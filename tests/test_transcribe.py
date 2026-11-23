"""Tests for Whisper segment post-processing and chunk splitting."""

from subgen.models import SubtitleSegment, WordTimestamp
from subgen.transcribe import postprocess_segments


def test_postprocess_filters_punctuation_and_hallucination_loops():
    segs = [
        SubtitleSegment(index=1, start=0.0, end=1.0, source_text="♪ ♫"),
        SubtitleSegment(index=2, start=1.0, end=2.0, source_text="ありがとう"),
        SubtitleSegment(index=3, start=2.0, end=3.0, source_text="ありがとう"),
        SubtitleSegment(index=4, start=3.0, end=4.0, source_text="ありがとう"),
        SubtitleSegment(index=5, start=4.0, end=5.0, source_text="ありがとう"),
        SubtitleSegment(index=6, start=5.0, end=6.5, source_text="今日はいい天気ですね。"),
    ]

    result = postprocess_segments(segs, language="ja")
    texts = [s.source_text for s in result]
    # Pure punctuation removed; repeated loop capped at 2 occurrences.
    assert texts == ["ありがとう", "ありがとう", "今日はいい天気ですね。"]
    assert [s.index for s in result] == [1, 2, 3]


def test_postprocess_splits_long_segment_using_word_timestamps():
    words = [
        WordTimestamp(word="Today", start=0.0, end=0.5),
        WordTimestamp(word=" we", start=0.5, end=0.8),
        WordTimestamp(word=" are", start=0.8, end=1.1),
        WordTimestamp(word=" building", start=1.1, end=1.6),
        WordTimestamp(word=" a", start=1.6, end=1.8),
        WordTimestamp(word=" fast", start=1.8, end=2.2),
        WordTimestamp(word=" subtitle", start=2.2, end=2.8),
        WordTimestamp(word=" tool.", start=2.8, end=3.4),
        WordTimestamp(word=" It", start=4.5, end=4.8),
        WordTimestamp(word=" supports", start=4.8, end=5.3),
        WordTimestamp(word=" both", start=5.3, end=5.7),
        WordTimestamp(word=" Japanese", start=5.7, end=6.4),
        WordTimestamp(word=" and", start=6.4, end=6.8),
        WordTimestamp(word=" English", start=6.8, end=7.5),
        WordTimestamp(word=" videos.", start=7.5, end=8.2),
    ]
    seg = SubtitleSegment(
        index=1,
        start=0.0,
        end=8.2,
        source_text="Today we are building a fast subtitle tool. It supports both Japanese and English videos.",
        words=words,
    )

    result = postprocess_segments([seg], language="en", max_duration=5.0, max_chars_en=50)
    assert len(result) == 2
    assert result[0].source_text == "Today we are building a fast subtitle tool."
    assert result[1].source_text == "It supports both Japanese and English videos."
    assert result[0].end <= result[1].start


def test_postprocess_fixes_overlapping_timestamps():
    segs = [
        SubtitleSegment(index=1, start=1.0, end=3.5, source_text="First line"),
        SubtitleSegment(index=2, start=3.0, end=5.0, source_text="Second line"),
    ]
    result = postprocess_segments(segs, language="en")
    assert result[0].end == 3.0
    assert result[1].start == 3.0
