"""Core data models for SubGen."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceLanguage = Literal["auto", "ja", "en"]
ASRBackendName = Literal["auto", "mlx", "faster-whisper", "openvino"]
TranslatorName = Literal["gemini", "openai"]
PipelineStage = Literal[
    "preparing",
    "extracting",
    "transcribing",
    "translating",
    "writing",
    "completed",
    "error",
]


class WordTimestamp(BaseModel):
    """Word-level timing emitted by Whisper."""

    word: str
    start: float
    end: float


class SubtitleSegment(BaseModel):
    """A single timed subtitle cue."""

    index: int
    start: float
    end: float
    source_text: str
    translated_text: str = ""
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Output of the local speech-to-text stage."""

    language: str
    segments: list[SubtitleSegment]
    duration: float | None = None


class PipelineConfig(BaseModel):
    """Configuration for a subtitle generation run."""

    source_lang: SourceLanguage = "auto"
    asr_backend: ASRBackendName = "auto"
    whisper_model: str = "auto"
    translator: TranslatorName = "openai"
    translation_model: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    batch_size: int = Field(default=30, ge=1, le=100)
    context_window: int = Field(default=5, ge=0, le=20)
    context_prompt: str = ""
    save_source_srt: bool = False
    bilingual: bool = False
    max_segment_duration: float = Field(default=7.0, gt=0)
    max_segment_chars_ja: int = Field(default=28, gt=0)
    max_segment_chars_en: int = Field(default=84, gt=0)


class ProgressEvent(BaseModel):
    """Progress update emitted during pipeline execution."""

    stage: PipelineStage
    progress: float = Field(ge=0.0, le=100.0)
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Final output returned when a pipeline run completes."""

    input_path: str
    output_srt_path: str
    source_srt_path: str | None = None
    detected_language: str
    segments: list[SubtitleSegment]
    duration: float | None = None
