"""Local Whisper ASR backends (MLX for Apple Silicon, faster-whisper for NVIDIA CUDA/CPU, OpenVINO for Intel iGPU)."""

from __future__ import annotations

from contextlib import contextmanager
import platform
import re
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Protocol

from subgen.audio import load_wav_as_float32
from subgen.config import (
    configure_model_cache,
    resolve_faster_whisper_model,
    resolve_mlx_model_repo,
    resolve_openvino_model_repo,
)
from subgen.hardware import (
    ASRPlan,
    HardwareProfile,
    detect_hardware,
    has_module,
    resolve_asr_plan,
)
from subgen.models import (
    ASRBackendName,
    SourceLanguage,
    SubtitleSegment,
    TranscriptionResult,
    WordTimestamp,
)

ProgressCallback = Callable[[float, str], None]

_JA_BREAK_PUNCT = set("。？！?!、，,")
_EN_BREAK_PUNCT = set(".?!;:,")
_EMPTY_PUNCT_RE = re.compile(r"^[\s\W_♪♫★☆・…ー\-]+$", re.UNICODE)
_MLX_TQDM_LOCK = threading.Lock()


def _format_clock(seconds: float) -> str:
    """Format duration in seconds as `MM:SS` or `H:MM:SS`."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _make_hf_download_tqdm(repo_id: str, on_progress: ProgressCallback | None):
    """Build a tqdm_class for `huggingface_hub.snapshot_download` that forwards download progress."""
    try:
        from huggingface_hub.utils import tqdm as base_tqdm  # type: ignore[import-untyped]
    except Exception:
        from tqdm import tqdm as base_tqdm  # type: ignore[import-untyped]

    short_name = repo_id.split("/")[-1]
    state = {"last_emit": 0.0, "last_pct": -1}

    class _SubGenDownloadTqdm(base_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            self._is_byte_bar = bool(kwargs.get("unit_scale")) or str(kwargs.get("unit", "")).upper() == "B"
            self._subgen_total = float(kwargs.get("total") or 0.0)
            self._subgen_n = float(kwargs.get("initial") or 0.0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            res = super().update(n)
            self._subgen_n += float(n or 0.0)
            total = self._subgen_total or float(getattr(self, "total", 0) or 0)
            if on_progress and total > 0:
                now = time.monotonic()
                ratio = min(1.0, max(0.0, self._subgen_n / total))
                pct_int = int(ratio * 100)
                if pct_int != state["last_pct"] and (now - state["last_emit"] >= 0.25 or pct_int >= 100):
                    state["last_emit"] = now
                    state["last_pct"] = pct_int
                    # Map model download into 2%..14% of the ASR stage.
                    asr_pct = 2.0 + ratio * 12.0
                    if self._is_byte_bar:
                        done_mb = self._subgen_n / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        on_progress(
                            asr_pct,
                            f"Downloading Whisper model ({short_name}): {done_mb:.1f} / {total_mb:.1f} MB ({pct_int}%)",
                        )
                    else:
                        on_progress(
                            asr_pct,
                            f"Preparing Whisper model ({short_name}): {int(self._subgen_n)}/{int(total)} files ({pct_int}%)",
                        )
            return res

    return _SubGenDownloadTqdm


@contextmanager
def _capture_mlx_whisper_progress(
    duration: float,
    model_label: str,
    on_progress: ProgressCallback | None,
) -> Iterator[None]:
    """Intercept mlx_whisper's internal tqdm frame progress bar to emit live transcription progress."""
    if on_progress is None:
        yield
        return

    import mlx_whisper.transcribe as mlx_transcribe  # type: ignore[import-untyped]

    with _MLX_TQDM_LOCK:
        original_tqdm = mlx_transcribe.tqdm.tqdm
        state = {"last_emit": 0.0, "last_pct": -1}

        class _MLXProgressTqdm(original_tqdm):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                self._subgen_unit = str(kwargs.get("unit", ""))
                self._subgen_total = float(kwargs.get("total") or 0.0)
                self._subgen_n = 0.0
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                res = super().update(n)
                self._subgen_n += float(n or 0.0)
                if self._subgen_unit == "frames" and self._subgen_total > 0:
                    ratio = min(1.0, max(0.0, self._subgen_n / self._subgen_total))
                    pct_int = int(ratio * 100)
                    now = time.monotonic()
                    if pct_int != state["last_pct"] and (now - state["last_emit"] >= 0.2 or pct_int >= 100):
                        state["last_emit"] = now
                        state["last_pct"] = pct_int
                        total_sec = duration if duration > 0 else self._subgen_total / 100.0
                        done_sec = ratio * total_sec
                        # Map Whisper decoding 0..100% into ASR 15..98%.
                        asr_pct = min(98.0, 15.0 + ratio * 83.0)
                        on_progress(
                            asr_pct,
                            f"Whisper ({model_label} · MLX): {_format_clock(done_sec)} / {_format_clock(total_sec)} ({pct_int}%)",
                        )
                return res

        mlx_transcribe.tqdm.tqdm = _MLXProgressTqdm
        try:
            yield
        finally:
            mlx_transcribe.tqdm.tqdm = original_tqdm


class ASRBackend(Protocol):
    """Protocol implemented by local speech recognition backends."""

    name: str

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "auto",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult: ...


def is_apple_silicon() -> bool:
    """Return True when running on macOS arm64."""
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def resolve_asr_backend_name(requested: ASRBackendName = "auto") -> str:
    """Resolve `auto` to the best installed local Whisper backend for current hardware."""
    return resolve_asr_plan(requested_backend=requested, requested_model="auto").backend


def create_asr_backend(
    requested: ASRBackendName = "auto",
    *,
    requested_model: str = "auto",
    hardware: HardwareProfile | None = None,
) -> tuple[ASRBackend, ASRPlan]:
    """Resolve hardware + installed packages and instantiate the matching ASR backend."""
    plan = resolve_asr_plan(
        requested_backend=requested,
        requested_model=requested_model,
        hardware=hardware,
    )
    if plan.backend == "mlx":
        return MLXWhisperBackend(), plan
    if plan.backend == "openvino":
        return OpenVINOWhisperBackend(device=plan.device, compute_type=plan.compute_type), plan
    return FasterWhisperBackend(device=plan.device, compute_type=plan.compute_type), plan


class MLXWhisperBackend:
    """Local Whisper transcription accelerated on Apple Silicon via MLX."""

    name = "mlx"

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "large-v3-turbo",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        cache_root = configure_model_cache()
        import mlx_whisper  # type: ignore[import-untyped]

        resolved_model = "large-v3-turbo" if model_name == "auto" else model_name
        repo_id = resolve_mlx_model_repo(resolved_model)
        if on_progress:
            on_progress(5.0, f"Loading audio and MLX Whisper model ({repo_id})...")

        # Resolve into the configured custom cache directory explicitly so models always land there
        # and emit live Hugging Face download progress on first run.
        model_target = repo_id
        candidate_dir = Path(repo_id)
        if not (candidate_dir.exists() and candidate_dir.is_dir()):
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            dl_kwargs: dict = {
                "repo_id": repo_id,
                "tqdm_class": _make_hf_download_tqdm(repo_id, on_progress),
            }
            if cache_root is not None:
                dl_kwargs["cache_dir"] = str(cache_root / "hub")
            model_target = str(snapshot_download(**dl_kwargs))

        audio = load_wav_as_float32(wav_path)
        duration = float(len(audio)) / 16000.0 if len(audio) > 0 else 0.0

        kwargs: dict = {
            "path_or_hf_repo": model_target,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "verbose": False,
        }
        if source_lang != "auto":
            kwargs["language"] = source_lang

        if on_progress:
            on_progress(
                15.0,
                f"Whisper ({resolved_model} · MLX): 00:00 / {_format_clock(duration)} (0%)",
            )

        with _capture_mlx_whisper_progress(duration, resolved_model, on_progress):
            result = mlx_whisper.transcribe(audio, **kwargs)
        detected_lang = str(result.get("language") or (source_lang if source_lang != "auto" else "en"))
        raw_segments = result.get("segments") or []

        segments: list[SubtitleSegment] = []
        for idx, seg in enumerate(raw_segments, start=1):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            words = [
                WordTimestamp(
                    word=str(w.get("word", "")),
                    start=float(w.get("start", seg.get("start", 0.0))),
                    end=float(w.get("end", seg.get("end", 0.0))),
                )
                for w in (seg.get("words") or [])
                if str(w.get("word", "")).strip()
            ]
            segments.append(
                SubtitleSegment(
                    index=idx,
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    source_text=text,
                    words=words,
                )
            )

        if on_progress:
            on_progress(100.0, f"Transcription complete ({len(segments)} raw segments, lang={detected_lang})")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=duration,
        )


class FasterWhisperBackend:
    """Local Whisper transcription via faster-whisper (CTranslate2) for NVIDIA CUDA dGPU & CPU."""

    name = "faster-whisper"

    def __init__(self, device: str = "auto", compute_type: str = "default") -> None:
        self.device = device
        self.compute_type = compute_type

    def _load_model(self, model_id: str, on_progress: ProgressCallback | None = None):
        cache_root = configure_model_cache()
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        device = self.device
        compute_type = self.compute_type
        if device == "auto":
            hw = detect_hardware()
            if hw.primary_accelerator == "nvidia_cuda":
                device = "cuda"
                compute_type = hw.recommended_compute_type
            else:
                device = "cpu"
                compute_type = "int8"

        model_kwargs: dict = {"device": device, "compute_type": compute_type}
        if cache_root is not None:
            model_kwargs["download_root"] = str(cache_root / "hub")

        try:
            return WhisperModel(model_id, **model_kwargs), device, compute_type
        except Exception as exc:
            # On Windows, NVIDIA GPU may exist without cuBLAS/cuDNN DLLs installed; degrade cleanly to CPU int8.
            if device == "cuda":
                if on_progress:
                    on_progress(
                        8.0,
                        f"CUDA runtime unavailable ({exc}); falling back to CPU (int8)...",
                    )
                fallback_kwargs: dict = {"device": "cpu", "compute_type": "int8"}
                if cache_root is not None:
                    fallback_kwargs["download_root"] = str(cache_root / "hub")
                return WhisperModel(model_id, **fallback_kwargs), "cpu", "int8"
            raise

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "large-v3-turbo",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        resolved_model = resolve_faster_whisper_model("large-v3-turbo" if model_name == "auto" else model_name)
        if on_progress:
            on_progress(5.0, f"Loading faster-whisper model ({resolved_model}, {self.device}/{self.compute_type})...")

        model, active_device, active_compute = self._load_model(resolved_model, on_progress=on_progress)
        audio = load_wav_as_float32(wav_path)
        duration = float(len(audio)) / 16000.0 if len(audio) > 0 else 0.0

        kwargs: dict = {
            "word_timestamps": True,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if source_lang != "auto":
            kwargs["language"] = source_lang

        if on_progress:
            on_progress(
                15.0,
                f"Whisper ({resolved_model} · {active_device.upper()}): 00:00 / {_format_clock(duration)} (0%)",
            )

        seg_iter, info = model.transcribe(audio, **kwargs)
        detected_lang = str(getattr(info, "language", None) or (source_lang if source_lang != "auto" else "en"))
        total_duration = float(getattr(info, "duration", duration) or duration or 1.0)

        segments: list[SubtitleSegment] = []
        for idx, seg in enumerate(seg_iter, start=1):
            text = (seg.text or "").strip()
            if not text:
                continue
            words = [
                WordTimestamp(
                    word=str(w.word),
                    start=float(w.start),
                    end=float(w.end),
                )
                for w in (seg.words or [])
                if str(w.word).strip()
            ]
            segments.append(
                SubtitleSegment(
                    index=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    source_text=text,
                    words=words,
                )
            )
            if on_progress and total_duration > 0:
                ratio = min(1.0, max(0.0, float(seg.end) / total_duration))
                pct = min(98.0, 15.0 + ratio * 83.0)
                on_progress(
                    pct,
                    f"Whisper ({resolved_model} · {active_device.upper()}): {_format_clock(float(seg.end))} / {_format_clock(total_duration)} ({int(ratio * 100)}%)",
                )

        if on_progress:
            on_progress(100.0, f"Transcription complete ({len(segments)} raw segments, lang={detected_lang})")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=total_duration,
        )


class OpenVINOWhisperBackend:
    """Local Whisper transcription accelerated on Intel iGPU / Arc (and CPU) via OpenVINO GenAI."""

    name = "openvino"

    def __init__(self, device: str = "GPU", compute_type: str = "int8") -> None:
        self.device = device
        self.compute_type = compute_type

    def _resolve_local_model_dir(self, repo_or_path: str, on_progress: ProgressCallback | None = None) -> str:
        cache_root = configure_model_cache()
        candidate = Path(repo_or_path)
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

        dl_kwargs: dict = {
            "repo_id": repo_or_path,
            "tqdm_class": _make_hf_download_tqdm(repo_or_path, on_progress),
        }
        if cache_root is not None:
            dl_kwargs["cache_dir"] = str(cache_root / "hub")
        return str(snapshot_download(**dl_kwargs))

    def _create_pipeline(self, model_dir: str, on_progress: ProgressCallback | None = None):
        import openvino_genai as ov_genai  # type: ignore[import-untyped]

        devices_to_try = [self.device]
        if self.device.upper() != "CPU":
            devices_to_try.append("CPU")

        last_exc: Exception | None = None
        for dev in devices_to_try:
            try:
                try:
                    return ov_genai.WhisperPipeline(model_dir, dev, word_timestamps=True), dev
                except TypeError:
                    return ov_genai.WhisperPipeline(model_dir, dev), dev
            except Exception as exc:
                last_exc = exc
                if dev != "CPU" and on_progress:
                    on_progress(10.0, f"OpenVINO {dev} init failed ({exc}); falling back to CPU...")

        raise RuntimeError(f"Failed to initialize OpenVINO WhisperPipeline: {last_exc}")

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "large-v3-turbo",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        resolved_alias = "large-v3-turbo" if model_name == "auto" else model_name
        repo_id = resolve_openvino_model_repo(resolved_alias, compute_type=self.compute_type)
        if on_progress:
            on_progress(5.0, f"Preparing OpenVINO Whisper model ({repo_id})...")

        model_dir = self._resolve_local_model_dir(repo_id, on_progress=on_progress)
        pipe, active_device = self._create_pipeline(model_dir, on_progress=on_progress)

        audio = load_wav_as_float32(wav_path)
        duration = float(len(audio)) / 16000.0 if len(audio) > 0 else 0.0
        raw_speech = audio.tolist()

        gen_kwargs: dict = {
            "task": "transcribe",
            "return_timestamps": True,
        }
        if source_lang != "auto":
            gen_kwargs["language"] = f"<|{source_lang}|>"

        if on_progress:
            on_progress(
                18.0,
                f"Whisper ({resolved_alias} · Intel {active_device}): 00:00 / {_format_clock(duration)} (0%)",
            )

        # Emit incremental progress if this OpenVINO GenAI build supports streamer callbacks.
        if on_progress:
            token_state = {"chars": 0, "last_emit": 0.0}
            est_chars = max(40.0, duration * 10.0)

            def _ov_streamer(subword: str) -> bool:
                token_state["chars"] += len(subword or "")
                now = time.monotonic()
                if now - token_state["last_emit"] >= 0.4:
                    token_state["last_emit"] = now
                    ratio = min(0.95, token_state["chars"] / est_chars)
                    pct = min(96.0, 18.0 + ratio * 78.0)
                    on_progress(
                        pct,
                        f"Whisper ({resolved_alias} · Intel {active_device}): transcribing (~{int(ratio * 100)}%)",
                    )
                return False

            try:
                result = pipe.generate(raw_speech, streamer=_ov_streamer, **gen_kwargs)
            except TypeError:
                result = pipe.generate(raw_speech, **gen_kwargs)
        else:
            result = pipe.generate(raw_speech, **gen_kwargs)
        raw_chunks = getattr(result, "chunks", None) or []

        segments: list[SubtitleSegment] = []
        for idx, chunk in enumerate(raw_chunks, start=1):
            text = str(getattr(chunk, "text", "") or "").strip()
            if not text:
                continue
            start_ts = float(getattr(chunk, "start_ts", 0.0))
            end_ts = float(getattr(chunk, "end_ts", start_ts + 0.1))
            words: list[WordTimestamp] = []
            for w in (getattr(chunk, "words", None) or []):
                w_text = str(getattr(w, "word", "") or getattr(w, "text", ""))
                if not w_text.strip():
                    continue
                words.append(
                    WordTimestamp(
                        word=w_text,
                        start=float(getattr(w, "start_ts", getattr(w, "start", start_ts))),
                        end=float(getattr(w, "end_ts", getattr(w, "end", end_ts))),
                    )
                )
            segments.append(
                SubtitleSegment(
                    index=idx,
                    start=start_ts,
                    end=max(end_ts, start_ts + 0.1),
                    source_text=text,
                    words=words,
                )
            )

        # Fallback if chunks were not populated by a specific OpenVINO build.
        if not segments:
            texts = getattr(result, "texts", None) or []
            full_text = str(texts[0] if texts else result).strip()
            if full_text:
                segments.append(
                    SubtitleSegment(
                        index=1,
                        start=0.0,
                        end=max(duration, 1.0),
                        source_text=full_text,
                    )
                )

        if source_lang != "auto":
            detected_lang = source_lang
        else:
            raw_lang = getattr(result, "language", None)
            if raw_lang:
                detected_lang = str(raw_lang).strip("<>| ").lower()
            else:
                detected_lang = _detect_language_from_text("".join(s.source_text for s in segments[:25]))

        if on_progress:
            on_progress(100.0, f"Transcription complete ({len(segments)} raw segments, lang={detected_lang})")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=duration,
        )


def _detect_language_from_text(sample: str) -> str:
    """Heuristic to distinguish Japanese vs English when an ASR backend omits language metadata."""
    kana_count = sum(1 for ch in sample if "\u3040" <= ch <= "\u30ff")
    return "ja" if kana_count >= 3 else "en"


def _join_words(words: list[WordTimestamp], is_cjk: bool) -> str:
    """Reconstruct natural text from Whisper word tokens."""
    if is_cjk:
        raw = "".join(w.word for w in words).strip()
        return re.sub(r"\s{2,}", " ", raw)

    # Whisper word tokens typically retain leading spaces for English.
    raw = "".join(w.word for w in words).strip()
    if raw and not any(w.word.startswith(" ") for w in words[1:]):
        raw = " ".join(w.word.strip() for w in words if w.word.strip())
    return re.sub(r"\s{2,}", " ", raw)


def _split_segment_with_words(
    seg: SubtitleSegment,
    *,
    is_cjk: bool,
    max_duration: float,
    max_chars: int,
    pause_threshold: float = 0.55,
) -> list[SubtitleSegment]:
    """Split an overly long segment into natural subtitle chunks using word timestamps."""
    text = seg.source_text.strip()
    duration = max(0.0, seg.end - seg.start)
    if not seg.words or (duration <= max_duration and len(text) <= max_chars):
        return [seg]

    break_punct = _JA_BREAK_PUNCT if is_cjk else _EN_BREAK_PUNCT
    chunks: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []

    for i, word in enumerate(seg.words):
        current.append(word)
        chunk_text = _join_words(current, is_cjk=is_cjk)
        chunk_dur = word.end - current[0].start
        next_word = seg.words[i + 1] if i + 1 < len(seg.words) else None
        gap_to_next = (next_word.start - word.end) if next_word is not None else 0.0

        ends_with_punct = bool(word.word.strip()) and word.word.strip()[-1] in break_punct
        hit_hard_limit = len(chunk_text) >= max_chars or chunk_dur >= max_duration
        hit_soft_break = (
            (len(chunk_text) >= max_chars * 0.65 or chunk_dur >= max_duration * 0.65)
            and (ends_with_punct or gap_to_next >= pause_threshold)
        )
        long_pause = gap_to_next >= max(pause_threshold * 1.6, 0.9) and len(current) >= 2

        if next_word is not None and (hit_hard_limit or hit_soft_break or long_pause):
            chunks.append(current)
            current = []

    if current:
        # Merge a tiny trailing orphan word back into the previous chunk if possible.
        if chunks and len(_join_words(current, is_cjk=is_cjk)) <= (4 if is_cjk else 10):
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    output: list[SubtitleSegment] = []
    for chunk in chunks:
        chunk_text = _join_words(chunk, is_cjk=is_cjk)
        if not chunk_text:
            continue
        output.append(
            SubtitleSegment(
                index=0,
                start=chunk[0].start,
                end=max(chunk[-1].end, chunk[0].start + 0.1),
                source_text=chunk_text,
                words=chunk,
            )
        )
    return output or [seg]


def postprocess_segments(
    segments: list[SubtitleSegment],
    language: str,
    *,
    max_duration: float = 7.0,
    max_chars_ja: int = 28,
    max_chars_en: int = 84,
) -> list[SubtitleSegment]:
    """Clean, filter repetitive Whisper hallucinations, split long cues, and normalize timing."""
    is_cjk = language.lower().startswith("ja") or language.lower().startswith("zh")
    max_chars = max_chars_ja if is_cjk else max_chars_en

    cleaned: list[SubtitleSegment] = []
    prev_text = ""
    repeat_count = 0

    for seg in segments:
        text = re.sub(r"\s+", " ", seg.source_text).strip()
        if not text or _EMPTY_PUNCT_RE.match(text):
            continue

        # Suppress Whisper hallucination loops where identical short lines repeat 3+ times in a row.
        normalized_key = re.sub(r"\s+", "", text.lower())
        if normalized_key == prev_text:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            prev_text = normalized_key
            repeat_count = 0

        candidate = seg.model_copy(update={"source_text": text})
        split_parts = _split_segment_with_words(
            candidate,
            is_cjk=is_cjk,
            max_duration=max_duration,
            max_chars=max_chars,
        )
        cleaned.extend(split_parts)

    # Normalize indices and ensure monotonically non-overlapping timestamps.
    normalized: list[SubtitleSegment] = []
    for seg in cleaned:
        start = max(0.0, seg.start)
        end = max(seg.end, start + 0.1)
        if normalized and start < normalized[-1].end:
            # Trim tiny overlap cleanly without collapsing the previous cue.
            if start > normalized[-1].start + 0.15:
                normalized[-1] = normalized[-1].model_copy(update={"end": start})
            else:
                start = normalized[-1].end
                end = max(end, start + 0.1)
        normalized.append(
            seg.model_copy(
                update={
                    "index": len(normalized) + 1,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
        )

    return normalized
