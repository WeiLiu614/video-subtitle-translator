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
    "large-v3-turbo-int4": "mlx-community/whisper-large-v3-turbo",
}

# Official OpenVINO IR Hugging Face repos for Intel iGPU / Arc / CPU on Windows & Linux.
OPENVINO_WHISPER_MODELS: dict[str, str] = {
    "tiny": "OpenVINO/whisper-tiny-fp16-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "large-v3-turbo-int4": "OpenVINO/whisper-large-v3-turbo-int4-ov",
}

# Canonical model aliases accepted by faster-whisper (CTranslate2).
FASTER_WHISPER_MODELS: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "large-v3-turbo-int4": "large-v3-turbo",
}

WHISPER_MODEL_CHOICES: list[str] = [
    "auto",
    "large-v3-turbo",
    "large-v3",
    "large-v3-turbo-int4",
    "medium",
    "small",
    "base",
    "tiny",
]

DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
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
    default_whisper_model: str = "auto"
    default_translator: str = "openai"
    default_gemini_model: str = DEFAULT_GEMINI_MODEL
    default_openai_model: str = DEFAULT_OPENAI_MODEL
    model_cache_dir: str | None = None


def load_settings(env_file: Path | None = None) -> AppSettings:
    """Load settings from `.env` (if present) and process environment."""
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    openai_key = os.getenv("OPENAI_API_KEY") or None
    openai_base_url = os.getenv("OPENAI_BASE_URL") or None
    model_cache_dir = os.getenv("SUBGEN_MODEL_CACHE_DIR") or os.getenv("HF_HOME") or None

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
        default_whisper_model=os.getenv("SUBGEN_WHISPER_MODEL", "auto"),
        default_translator=default_translator,
        default_gemini_model=os.getenv("SUBGEN_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        default_openai_model=os.getenv("SUBGEN_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        model_cache_dir=model_cache_dir,
    )


def get_model_cache_dir(cache_dir: str | Path | None = None) -> Path | None:
    """Return the configured local model cache root directory, or None for default (~/.cache/huggingface)."""
    raw = str(cache_dir).strip() if cache_dir is not None else (load_settings().model_cache_dir or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def configure_model_cache(cache_dir: str | Path | None = None) -> Path | None:
    """Configure Hugging Face cache env vars (HF_HOME, HF_HUB_CACHE, HF_XET_CACHE) to store Whisper models."""
    root = get_model_cache_dir(cache_dir)
    if root is None:
        return None

    hub_dir = root / "hub"
    xet_dir = root / "xet"
    hub_dir.mkdir(parents=True, exist_ok=True)
    xet_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_XET_CACHE"] = str(xet_dir)
    return root


def resolve_mlx_model_repo(model_name: str) -> str:
    """Resolve a short Whisper model alias into an MLX Hugging Face repo ID."""
    cleaned = model_name.strip()
    if "/" in cleaned:
        return cleaned
    return MLX_WHISPER_MODELS.get(cleaned, cleaned)


def resolve_openvino_model_repo(model_name: str, compute_type: str = "int8") -> str:
    """Resolve a short Whisper model alias into an OpenVINO Hugging Face repo ID."""
    cleaned = model_name.strip()
    if "/" in cleaned:
        return cleaned
    if cleaned == "large-v3-turbo" and compute_type == "int4":
        return OPENVINO_WHISPER_MODELS["large-v3-turbo-int4"]
    return OPENVINO_WHISPER_MODELS.get(cleaned, cleaned)


def resolve_faster_whisper_model(model_name: str) -> str:
    """Resolve a short Whisper model alias into a CTranslate2 / faster-whisper model name."""
    cleaned = model_name.strip()
    if "/" in cleaned:
        return cleaned
    return FASTER_WHISPER_MODELS.get(cleaned, cleaned)
