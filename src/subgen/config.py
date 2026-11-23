"""Environment configuration and model presets for SubGen."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Standard MLX Hugging Face repos for Whisper models on Apple Silicon.
MLX_WHISPER_MODELS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
    ".ts",
    ".mpg",
    ".mpeg",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}

SUPPORTED_SUBTITLE_EXTENSIONS = {".srt"}


@dataclass
class AppSettings:
    """Resolved settings loaded from environment variables and .env."""

    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    default_asr_backend: str = "auto"
    default_whisper_model: str = "large-v3-turbo"
    default_translator: str = "openai"
    default_gemini_model: str = DEFAULT_GEMINI_MODEL
    default_openai_model: str = DEFAULT_OPENAI_MODEL


def load_settings(env_file: Path | None = None) -> AppSettings:
    """Load settings from `.env` (if present) and process environment."""
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    openai_key = os.getenv("OPENAI_API_KEY") or None
    openai_base_url = os.getenv("OPENAI_BASE_URL") or None

    default_translator = os.getenv("SUBGEN_TRANSLATOR")
    if not default_translator:
        if openai_key:
            default_translator = "openai"
        elif gemini_key:
            default_translator = "gemini"
        else:
            default_translator = "openai"

    return AppSettings(
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        openai_base_url=openai_base_url,
        default_asr_backend=os.getenv("SUBGEN_ASR_BACKEND", "auto"),
        default_whisper_model=os.getenv("SUBGEN_WHISPER_MODEL", "large-v3-turbo"),
        default_translator=default_translator,
        default_gemini_model=os.getenv("SUBGEN_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        default_openai_model=os.getenv("SUBGEN_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    )


def resolve_mlx_model_repo(model_name: str) -> str:
    """Resolve a short Whisper model alias into an MLX Hugging Face repo ID."""
    cleaned = model_name.strip()
    if "/" in cleaned:
        return cleaned
    return MLX_WHISPER_MODELS.get(cleaned, cleaned)
