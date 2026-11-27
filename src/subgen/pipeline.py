"""End-to-end subtitle generation pipeline shared by CLI and Web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from subgen.audio import extract_audio_wav, probe_duration
from subgen.config import SUPPORTED_SUBTITLE_EXTENSIONS
from subgen.models import (
    PipelineConfig,
    PipelineResult,
    ProgressEvent,
    SubtitleSegment,
)
from subgen.srt import read_srt, write_srt
from subgen.transcribe import ASRBackend, create_asr_backend, postprocess_segments
from subgen.translate import LLMClient, SubtitleTranslator, create_llm_client

EventCallback = Callable[[ProgressEvent], None]


def default_output_srt_path(input_path: Path, bilingual: bool = False) -> Path:
    """Compute a clean default output SRT path beside the input video/audio/SRT file."""
    suffix = ".bilingual.srt" if bilingual else ".zh.srt"
    return input_path.with_name(f"{input_path.stem}{suffix}")


def default_source_srt_path(input_path: Path, language: str) -> Path:
    """Compute default path for saving the raw source-language SRT."""
    lang_tag = (language or "src").lower().split("-")[0]
    return input_path.with_name(f"{input_path.stem}.{lang_tag}.srt")


def run_pipeline(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    on_event: EventCallback | None = None,
    asr_backend: ASRBackend | None = None,
    llm_client: LLMClient | None = None,
) -> PipelineResult:
    """Run the full video/audio/SRT -> Simplified Chinese SRT pipeline."""
    cfg = config or PipelineConfig()
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    out_srt = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_output_srt_path(src, bilingual=cfg.bilingual)
    )

    def emit(stage, progress: float, message: str, detail: dict | None = None) -> None:
        if on_event:
            on_event(
                ProgressEvent(
                    stage=stage,
                    progress=round(min(100.0, max(0.0, progress)), 1),
                    message=message,
                    detail=detail or {},
                )
            )

    emit("preparing", 2.0, f"Preparing {src.name}...")

    is_srt_input = src.suffix.lower() in SUPPORTED_SUBTITLE_EXTENSIONS
    duration: float | None = None
    detected_lang = cfg.source_lang if cfg.source_lang != "auto" else "ja"
    segments: list[SubtitleSegment] = []

    if is_srt_input:
        emit("transcribing", 15.0, f"Reading existing subtitle file {src.name}...")
        segments = read_srt(src)
        if not segments:
            raise ValueError(f"No subtitle cues found in {src.name}")
        detected_lang = _guess_language_from_segments(segments) if cfg.source_lang == "auto" else cfg.source_lang
        emit("transcribing", 50.0, f"Loaded {len(segments)} cues from SRT (lang={detected_lang})")
    else:
        duration = probe_duration(src)
        emit("extracting", 8.0, f"Extracting 16kHz audio track from {src.name}...")
        wav_path = extract_audio_wav(src)
        try:
            if asr_backend is not None:
                backend = asr_backend
                resolved_model = "large-v3-turbo" if cfg.whisper_model == "auto" else cfg.whisper_model
                plan_desc = f"{backend.name}, model={resolved_model}"
            else:
                backend, plan = create_asr_backend(
                    cfg.asr_backend,
                    requested_model=cfg.whisper_model,
                )
                resolved_model = plan.model_name
                plan_desc = f"{plan.backend} on {plan.device} ({plan.compute_type}), model={resolved_model}"

            emit("transcribing", 15.0, f"Starting local Whisper ({plan_desc})...")

            def on_asr_progress(pct: float, msg: str) -> None:
                # Map ASR 0..100% into overall 15..55%, while exposing raw stage_progress (0..100%)
                emit("transcribing", 15.0 + pct * 0.40, msg, {"stage_progress": round(pct, 1)})

            asr_result = backend.transcribe(
                wav_path,
                source_lang=cfg.source_lang,
                model_name=resolved_model,
                on_progress=on_asr_progress,
            )
            detected_lang = asr_result.language
            if asr_result.duration is not None:
                duration = asr_result.duration

            segments = postprocess_segments(
                asr_result.segments,
                detected_lang,
                max_duration=cfg.max_segment_duration,
                max_chars_ja=cfg.max_segment_chars_ja,
                max_chars_en=cfg.max_segment_chars_en,
            )
            emit(
                "transcribing",
                55.0,
                f"Transcribed {len(segments)} subtitle cues (detected language: {detected_lang})",
                {"detected_language": detected_lang, "segment_count": len(segments)},
            )
        finally:
            wav_path.unlink(missing_ok=True)

    if not segments:
        raise RuntimeError("No speech was detected in the input media.")

    source_srt_path: Path | None = None
    if cfg.save_source_srt and not is_srt_input:
        source_srt_path = write_srt(
            segments,
            default_source_srt_path(out_srt.with_name(src.name), detected_lang),
            mode="source",
        )

    client = llm_client or create_llm_client(
        provider=cfg.translator,
        model=cfg.translation_model,
        api_key=cfg.api_key,
        base_url=cfg.api_base_url,
    )
    translator = SubtitleTranslator(
        client,
        batch_size=cfg.batch_size,
        context_window=cfg.context_window,
    )

    def on_translate_progress(pct: float, msg: str) -> None:
        # Map translation 0..100% into overall 55..95%, while exposing raw stage_progress (0..100%)
        emit("translating", 55.0 + pct * 0.40, msg, {"stage_progress": round(pct, 1)})

    emit("translating", 56.0, f"Translating {len(segments)} cues to Simplified Chinese...")
    translated_segments = translator.translate_segments(
        segments,
        source_language=detected_lang,
        context_prompt=cfg.context_prompt,
        on_progress=on_translate_progress,
    )

    emit("writing", 96.0, f"Writing SRT file to {out_srt.name}...")
    srt_mode = "bilingual" if cfg.bilingual else "translated"
    write_srt(translated_segments, out_srt, mode=srt_mode)

    emit(
        "completed",
        100.0,
        f"Done! Generated {len(translated_segments)} Chinese subtitles -> {out_srt.name}",
        {"output_srt_path": str(out_srt)},
    )

    return PipelineResult(
        input_path=str(src),
        output_srt_path=str(out_srt),
        source_srt_path=str(source_srt_path) if source_srt_path else None,
        detected_language=detected_lang,
        segments=translated_segments,
        duration=duration,
    )


def _guess_language_from_segments(segments: list[SubtitleSegment]) -> str:
    """Heuristic to detect Japanese vs English when translating an existing SRT."""
    sample = "".join(seg.source_text for seg in segments[:25])
    # Hiragana (3040-309F) or Katakana (30A0-30FF) strongly indicates Japanese.
    kana_count = sum(1 for ch in sample if "\u3040" <= ch <= "\u30ff")
    return "ja" if kana_count >= 3 else "en"
