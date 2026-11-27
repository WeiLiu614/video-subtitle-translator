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


def test_format_clock_and_whisper_progress_hooks():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    import tqdm

    from subgen.transcribe import (
        _capture_mlx_whisper_progress,
        _format_clock,
        _make_hf_download_tqdm,
    )

    assert _format_clock(0) == "00:00"
    assert _format_clock(75.4) == "01:15"
    assert _format_clock(3665) == "1:01:05"

    # Test HF download progress tqdm wrapper
    dl_events: list[tuple[float, str]] = []
    dl_tqdm_cls = _make_hf_download_tqdm(
        "mlx-community/whisper-large-v3-turbo",
        lambda pct, msg: dl_events.append((pct, msg)),
    )
    bar = dl_tqdm_cls(total=10 * 1024 * 1024, unit="B", unit_scale=True)
    bar.update(5 * 1024 * 1024)
    bar.update(5 * 1024 * 1024)
    bar.close()
    assert len(dl_events) >= 1
    assert "Downloading Whisper model (whisper-large-v3-turbo)" in dl_events[-1][1]
    assert "(100%)" in dl_events[-1][1]

    # Test MLX Whisper frame progress interception & restoration
    mlx_events: list[tuple[float, str]] = []
    fake_tqdm_mod = SimpleNamespace(tqdm=tqdm.tqdm)
    fake_mlx_transcribe = SimpleNamespace(tqdm=fake_tqdm_mod)
    fake_mlx_pkg = SimpleNamespace(transcribe=fake_mlx_transcribe)

    with patch.dict(
        sys.modules,
        {"mlx_whisper": fake_mlx_pkg, "mlx_whisper.transcribe": fake_mlx_transcribe},
    ):
        with _capture_mlx_whisper_progress(
            duration=120.0,
            model_label="large-v3-turbo",
            on_progress=lambda pct, msg: mlx_events.append((pct, msg)),
        ):
            with fake_mlx_transcribe.tqdm.tqdm(total=12000, unit="frames", disable=False) as pbar:
                pbar.update(6000)
                pbar.update(6000)

        # Verify original tqdm.tqdm was restored cleanly
        assert fake_mlx_transcribe.tqdm.tqdm is tqdm.tqdm

    assert len(mlx_events) >= 1
    assert "Whisper (large-v3-turbo · MLX): 02:00 / 02:00 (100%)" in mlx_events[-1][1]
    assert mlx_events[-1][0] == 98.0

