"""Tests for the FastAPI Web UI endpoints."""

import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from subgen.models import PipelineResult, SubtitleSegment
from subgen.srt import write_srt
from subgen.web.app import create_app


def test_status_and_index(tmp_path: Path):
    app = create_app(upload_root=tmp_path)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert "SubGen" in res.text

    status = client.get("/api/status").json()
    assert "whisper_models" in status
    assert "auto" in status["whisper_models"]
    assert "asr_backends" in status
    assert "openvino" in status["asr_backends"]
    assert "hardware" in status
    assert "recommended_backend" in status["hardware"]


def test_job_lifecycle_edit_and_download(tmp_path: Path):
    app = create_app(upload_root=tmp_path)
    client = TestClient(app)

    def fake_run_pipeline(input_path, output_path=None, config=None, on_event=None, **kwargs):
        segs = [
            SubtitleSegment(
                index=1,
                start=0.0,
                end=2.0,
                source_text="Hello there",
                translated_text="你好呀",
            )
        ]
        out = Path(output_path) if output_path else Path(input_path).with_suffix(".zh.srt")
        write_srt(segs, out, mode="translated")
        return PipelineResult(
            input_path=str(input_path),
            output_srt_path=str(out),
            detected_language="en",
            segments=segs,
            duration=2.0,
        )

    with patch("subgen.web.app.run_pipeline", side_effect=fake_run_pipeline):
        upload_res = client.post(
            "/api/jobs",
            files={"file": ("clip.srt", b"1\n00:00:00,000 --> 00:00:02,000\nHello there\n", "text/plain")},
            data={"source_lang": "en", "translator": "gemini"},
        )
        assert upload_res.status_code == 200
        job_id = upload_res.json()["job_id"]

        # Wait briefly for background worker thread
        for _ in range(30):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.05)

        assert job["status"] == "completed"
        assert job["segments"][0]["translated_text"] == "你好呀"

        # Edit subtitle via PUT
        edit_res = client.put(
            f"/api/jobs/{job_id}/segments",
            json={"segments": [{"index": 1, "translated_text": "嗨，你好！"}]},
        )
        assert edit_res.status_code == 200

        # Check WebVTT preview reflects edit
        vtt_res = client.get(f"/api/jobs/{job_id}/subtitles.vtt?mode=translated")
        assert vtt_res.status_code == 200
        assert "嗨，你好！" in vtt_res.text

        # Check SRT download reflects edit
        dl_res = client.get(f"/api/jobs/{job_id}/download?mode=translated")
        assert dl_res.status_code == 200
        assert "嗨，你好！" in dl_res.text
        assert ".zh.srt" in dl_res.headers["content-disposition"]
