"""Local Whisper ASR backends (MLX Whisper for Apple Silicon & faster-whisper) and segment cleanup."""

from __future__ import annotations

import importlib.util
import platform
import re
from pathlib import Path
from typing import Callable, Protocol

from subgen.audio import load_wav_as_float32
from subgen.config import resolve_mlx_model_repo
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


class ASRBackend(Protocol):
    """Protocol implemented by local speech recognition backends."""

    name: str

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "large-v3-turbo",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult: ...


def is_apple_silicon() -> bool:
    """Return True when running on macOS arm64."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_module(module_name: str) -> bool:
    """Return True if a Python package is importable."""
    return importlib.util.find_spec(module_name) is not None


def resolve_asr_backend_name(requested: ASRBackendName = "auto") -> str:
    """Resolve `auto` to the best installed local Whisper backend."""
    if requested == "mlx":
        if not has_module("mlx_whisper"):
            raise RuntimeError(
                "mlx-whisper is not installed. Install it with: pip install 'subgen[mlx]' (or pip install mlx-whisper)"
            )
        return "mlx"

    if requested == "faster-whisper":
        if not has_module("faster_whisper"):
            raise RuntimeError(
                "faster-whisper is not installed. Install it with: pip install 'subgen[faster-whisper]'"
            )
        return "faster-whisper"

    # auto mode
    if is_apple_silicon() and has_module("mlx_whisper"):
        return "mlx"
    if has_module("faster_whisper"):
        return "faster-whisper"
    if has_module("mlx_whisper"):
        return "mlx"

    raise RuntimeError(
        "No local Whisper backend is installed. Install `mlx-whisper` (recommended on Apple Silicon) "
        "or `faster-whisper`."
    )


def create_asr_backend(requested: ASRBackendName = "auto") -> ASRBackend:
    """Instantiate the selected ASR backend."""
    resolved = resolve_asr_backend_name(requested)
    if resolved == "mlx":
        return MLXWhisperBackend()
    return FasterWhisperBackend()


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
        import mlx_whisper  # type: ignore[import-untyped]

        repo_id = resolve_mlx_model_repo(model_name)
        if on_progress:
            on_progress(5.0, f"Loading audio and MLX Whisper model ({repo_id})...")

        audio = load_wav_as_float32(wav_path)
        duration = float(len(audio)) / 16000.0 if len(audio) > 0 else 0.0

        kwargs: dict = {
            "path_or_hf_repo": repo_id,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "verbose": False,
        }
        if source_lang != "auto":
            kwargs["language"] = source_lang

        if on_progress:
            on_progress(15.0, f"Transcribing with MLX Whisper ({model_name})...")

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
    """Local Whisper transcription via faster-whisper (CTranslate2) with VAD."""

    name = "faster-whisper"

    def transcribe(
        self,
        wav_path: Path,
        source_lang: SourceLanguage = "auto",
        model_name: str = "large-v3-turbo",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        if on_progress:
            on_progress(5.0, f"Loading faster-whisper model ({model_name})...")

        model = WhisperModel(model_name, device="auto", compute_type="default")
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
            on_progress(15.0, f"Transcribing with faster-whisper ({model_name})...")

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
                pct = min(98.0, 15.0 + (float(seg.end) / total_duration) * 83.0)
                on_progress(pct, f"Transcribed {seg.end:.1f}s / {total_duration:.1f}s")

        if on_progress:
            on_progress(100.0, f"Transcription complete ({len(segments)} raw segments, lang={detected_lang})")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=total_duration,
        )


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
