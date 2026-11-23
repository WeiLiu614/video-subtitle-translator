"""Tests for SRT and WebVTT formatting and parsing."""

from pathlib import Path

from subgen.models import SubtitleSegment
from subgen.srt import (
    format_srt_timestamp,
    format_vtt_timestamp,
    parse_srt,
    read_srt,
    render_srt,
    render_vtt,
    write_srt,
)


def test_timestamp_formatting():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(65.432) == "00:01:05,432"
    assert format_srt_timestamp(3661.999) == "01:01:01,999"
    assert format_srt_timestamp(-2.0) == "00:00:00,000"
    assert format_vtt_timestamp(65.432) == "00:01:05.432"


def test_render_and_parse_roundtrip(tmp_path: Path):
    segments = [
        SubtitleSegment(
            index=1,
            start=0.5,
            end=2.75,
            source_text="おはようございます。",
            translated_text="早上好。",
        ),
        SubtitleSegment(
            index=2,
            start=3.0,
            end=5.2,
            source_text="Hello, welcome back!",
            translated_text="你好，欢迎回来！",
        ),
    ]

    srt_text = render_srt(segments, mode="translated")
    assert "00:00:00,500 --> 00:00:02,750\n早上好。" in srt_text
    assert "00:00:03,000 --> 00:00:05,200\n你好，欢迎回来！" in srt_text

    bilingual_text = render_srt(segments, mode="bilingual")
    assert "早上好。\nおはようございます。" in bilingual_text

    vtt_text = render_vtt(segments, mode="translated")
    assert vtt_text.startswith("WEBVTT")
    assert "00:00:00.500 --> 00:00:02.750" in vtt_text

    out_path = write_srt(segments, tmp_path / "sample.zh.srt")
    parsed = read_srt(out_path)
    assert len(parsed) == 2
    assert parsed[0].source_text == "早上好。"
    assert abs(parsed[0].start - 0.5) < 1e-3
    assert parsed[1].source_text == "你好，欢迎回来！"


def test_parse_srt_handles_bom_and_crlf():
    raw = "\ufeff1\r\n00:00:01,000 --> 00:00:02,500\r\nLine 1\r\nLine 2\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nSecond\r\n"
    segs = parse_srt(raw)
    assert len(segs) == 2
    assert segs[0].source_text == "Line 1\nLine 2"
    assert segs[1].start == 3.0
