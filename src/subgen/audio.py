"""Audio extraction and media probing utilities using FFmpeg."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class FFmpegNotFoundError(RuntimeError):
    """Raised when neither system ffmpeg nor imageio-ffmpeg is available."""


def find_ffmpeg() -> str:
    """Locate an executable ffmpeg binary (system PATH first, then imageio-ffmpeg)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except ImportError:
        pass

    raise FFmpegNotFoundError(
        "FFmpeg not found. Install it via `brew install ffmpeg` or `pip install imageio-ffmpeg`."
    )


def probe_duration(media_path: str | Path) -> float | None:
    """Return media duration in seconds using ffmpeg stderr output, or None if unavailable."""
    try:
        ffmpeg = find_ffmpeg()
    except FFmpegNotFoundError:
        return None

    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    match = _DURATION_RE.search(proc.stderr)
    if not match:
        return None

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_audio_wav(
    input_path: str | Path,
    output_wav: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Extract 16kHz mono 16-bit PCM WAV audio from any video/audio file."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input media file not found: {src}")

    ffmpeg = find_ffmpeg()
    if output_wav is None:
        tmp = tempfile.NamedTemporaryFile(prefix="subgen_", suffix=".wav", delete=False)
        tmp.close()
        dst = Path(tmp.name)
    else:
        dst = Path(output_wav)
        dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        if dst.exists() and output_wav is None:
            dst.unlink(missing_ok=True)
        stderr_msg = proc.stderr.strip() or f"ffmpeg exited with code {proc.returncode}"
        raise RuntimeError(f"Failed to extract audio from {src.name}: {stderr_msg}")

    return dst


def load_wav_as_float32(wav_path: str | Path):
    """Load a 16-bit mono PCM WAV file into a float32 numpy array normalized to [-1, 1]."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError(
                f"Expected 16-bit mono WAV, got {wf.getnchannels()}ch {wf.getsampwidth() * 8}-bit"
            )
        raw_frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio
