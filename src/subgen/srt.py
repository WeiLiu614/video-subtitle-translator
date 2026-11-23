"""SRT and WebVTT subtitle formatting, parsing, and writing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

from subgen.models import SubtitleSegment

_SRT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as `HH:MM:SS,mmm`."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as `HH:MM:SS.mmm` for HTML5 WebVTT tracks."""
    return format_srt_timestamp(seconds).replace(",", ".")


def parse_timestamp(hours: str, minutes: str, seconds: str, ms: str) -> float:
    """Convert parsed timestamp components into seconds."""
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000.0


def render_srt(
    segments: Iterable[SubtitleSegment],
    mode: Literal["translated", "source", "bilingual"] = "translated",
) -> str:
    """Render subtitle segments to standard SRT text."""
    blocks: list[str] = []
    cue_num = 1
    for seg in segments:
        start_ts = format_srt_timestamp(seg.start)
        end_ts = format_srt_timestamp(max(seg.end, seg.start + 0.05))

        if mode == "source":
            body = seg.source_text.strip()
        elif mode == "bilingual":
            zh = seg.translated_text.strip()
            src = seg.source_text.strip()
            if zh and src and zh != src:
                body = f"{zh}\n{src}"
            else:
                body = zh or src
        else:
            body = (seg.translated_text or seg.source_text).strip()

        if not body:
            continue

        blocks.append(f"{cue_num}\n{start_ts} --> {end_ts}\n{body}")
        cue_num += 1

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(
    segments: Iterable[SubtitleSegment],
    mode: Literal["translated", "source", "bilingual"] = "translated",
) -> str:
    """Render subtitle segments to WebVTT text for browser video preview."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start_ts = format_vtt_timestamp(seg.start)
        end_ts = format_vtt_timestamp(max(seg.end, seg.start + 0.05))

        if mode == "source":
            body = seg.source_text.strip()
        elif mode == "bilingual":
            zh = seg.translated_text.strip()
            src = seg.source_text.strip()
            body = f"{zh}\n{src}" if (zh and src and zh != src) else (zh or src)
        else:
            body = (seg.translated_text or seg.source_text).strip()

        if not body:
            continue

        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(body)
        lines.append("")

    return "\n".join(lines)


def write_srt(
    segments: Iterable[SubtitleSegment],
    output_path: str | Path,
    mode: Literal["translated", "source", "bilingual"] = "translated",
    utf8_bom: bool = False,
) -> Path:
    """Write subtitle segments to an SRT file on disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_srt(segments, mode=mode)
    encoding = "utf-8-sig" if utf8_bom else "utf-8"
    path.write_text(content, encoding=encoding)
    return path


def parse_srt(content: str) -> list[SubtitleSegment]:
    """Parse SRT file text into a list of SubtitleSegment objects."""
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n{2,}", normalized.strip())
    segments: list[SubtitleSegment] = []

    for block in raw_blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip() != ""]
        if len(lines) < 2:
            continue

        ts_line_idx = 0
        match = _SRT_TIMESTAMP_RE.search(lines[0])
        if match is None and len(lines) >= 2:
            ts_line_idx = 1
            match = _SRT_TIMESTAMP_RE.search(lines[1])
        if match is None:
            continue

        start = parse_timestamp(*match.groups()[:4])
        end = parse_timestamp(*match.groups()[4:])
        text = "\n".join(lines[ts_line_idx + 1 :]).strip()
        if not text:
            continue

        segments.append(
            SubtitleSegment(
                index=len(segments) + 1,
                start=start,
                end=max(end, start),
                source_text=text,
            )
        )

    return segments


def read_srt(path: str | Path) -> list[SubtitleSegment]:
    """Read and parse an SRT file from disk."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse_srt(text)
