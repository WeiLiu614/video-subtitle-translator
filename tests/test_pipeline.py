"""End-to-end pipeline tests with real synthetic audio via FFmpeg and mocked ASR/LLM."""

import json
import subprocess
from pathlib import Path

from subgen.audio import find_ffmpeg
from subgen.models import PipelineConfig, ProgressEvent, SubtitleSegment, TranscriptionResult
from subgen.pipeline import run_pipeline
from subgen.srt import read_srt


class MockASRBackend:
    name = "mock-asr"

    def transcribe(self, wav_path, source_lang="auto", model_name="large-v3-turbo", on_progress=None):
        assert wav_path.exists() and wav_path.stat().st_size > 0
        if on_progress:
            on_progress(100.0, "Mock ASR done")
        return TranscriptionResult(
            language="ja" if source_lang == "auto" else source_lang,
            duration=2.0,
            segments=[
                SubtitleSegment(index=1, start=0.1, end=1.0, source_text="こんにちは、世界！"),
                SubtitleSegment(index=2, start=1.1, end=1.9, source_text="よろしくお願いします。"),
            ],
        )


class MockLLMClient:
    name = "mock-llm"
    model = "mock-model"

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        translations = {
            "こんにちは、世界！": "你好，世界！",
            "よろしくお願いします。": "请多多关照。",
            "Hello world": "你好世界",
        }
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "text": translations.get(item["source"], item["source"])}
                    for item in payload["items"]
                ]
            },
            ensure_ascii=False,
        )


def test_pipeline_with_synthetic_mp4(tmp_path: Path):
    ffmpeg = find_ffmpeg()
    sample_mp4 = tmp_path / "demo.mp4"
    # Create a tiny 2-second silent MP4 video with audio track.
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:d=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(sample_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    events: list[ProgressEvent] = []
    result = run_pipeline(
        sample_mp4,
        config=PipelineConfig(save_source_srt=True),
        on_event=events.append,
        asr_backend=MockASRBackend(),
        llm_client=MockLLMClient(),
    )

    assert result.detected_language == "ja"
    out_srt = Path(result.output_srt_path)
    assert out_srt.exists()
    assert out_srt.name == "demo.zh.srt"
    assert result.source_srt_path and Path(result.source_srt_path).exists()

    parsed = read_srt(out_srt)
    assert [p.source_text for p in parsed] == ["你好，世界！", "请多多关照。"]
    assert events[-1].stage == "completed"
    assert events[-1].progress == 100.0


def test_pipeline_direct_srt_translation(tmp_path: Path):
    src_srt = tmp_path / "talk.en.srt"
    src_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello world\n", encoding="utf-8")

    result = run_pipeline(
        src_srt,
        llm_client=MockLLMClient(),
    )
    assert result.detected_language == "en"
    assert Path(result.output_srt_path).read_text(encoding="utf-8").strip().endswith("你好世界")
