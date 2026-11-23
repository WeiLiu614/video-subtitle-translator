"""FastAPI backend for the SubGen local Web UI."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from subgen import __version__
from subgen.audio import FFmpegNotFoundError, find_ffmpeg
from subgen.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    MLX_WHISPER_MODELS,
    SUPPORTED_SUBTITLE_EXTENSIONS,
    load_settings,
)
from subgen.models import PipelineConfig, ProgressEvent, SubtitleSegment
from subgen.pipeline import run_pipeline
from subgen.srt import render_srt, render_vtt, write_srt
from subgen.transcribe import has_module, is_apple_silicon

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_ROOT = Path.cwd() / ".subgen_uploads"


@dataclass
class JobRecord:
    """In-memory state for a Web UI subtitle generation job."""

    job_id: str
    filename: str
    input_path: Path
    output_srt_path: Path | None = None
    status: Literal["queued", "running", "completed", "error"] = "queued"
    stage: str = "preparing"
    progress: float = 0.0
    message: str = "Queued..."
    error: str | None = None
    detected_language: str | None = None
    duration: float | None = None
    segments: list[SubtitleSegment] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def is_media(self) -> bool:
        return self.input_path.suffix.lower() not in SUPPORTED_SUBTITLE_EXTENSIONS

    def push_event(self, event: ProgressEvent) -> None:
        with self.lock:
            self.stage = event.stage
            self.progress = event.progress
            self.message = event.message
            self.events.append(event.model_dump())

    def to_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "filename": self.filename,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
                "detected_language": self.detected_language,
                "duration": self.duration,
                "output_srt_path": str(self.output_srt_path) if self.output_srt_path else None,
                "media_available": self.is_media and self.input_path.exists(),
                "segments": [s.model_dump(exclude={"words"}) for s in self.segments],
            }


class SegmentUpdateItem(BaseModel):
    index: int
    translated_text: str


class SegmentsUpdatePayload(BaseModel):
    segments: list[SegmentUpdateItem]


def create_app(upload_root: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="SubGen Web UI", version=__version__)
    jobs: dict[str, JobRecord] = {}
    storage_root = upload_root or UPLOAD_ROOT

    app.state.jobs = jobs
    app.state.upload_root = storage_root

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Web UI assets not found.")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        settings = load_settings()
        try:
            ffmpeg_path = find_ffmpeg()
        except FFmpegNotFoundError:
            ffmpeg_path = None

        return {
            "version": __version__,
            "apple_silicon": is_apple_silicon(),
            "ffmpeg_available": ffmpeg_path is not None,
            "asr_backends": {
                "mlx": has_module("mlx_whisper"),
                "faster_whisper": has_module("faster_whisper"),
            },
            "whisper_models": list(MLX_WHISPER_MODELS.keys()),
            "defaults": {
                "asr_backend": settings.default_asr_backend,
                "whisper_model": settings.default_whisper_model,
                "translator": settings.default_translator,
                "gemini_model": settings.default_gemini_model or DEFAULT_GEMINI_MODEL,
                "openai_model": settings.default_openai_model or DEFAULT_OPENAI_MODEL,
                "openai_base_url": settings.openai_base_url or "",
            },
            "api_keys_configured": {
                "gemini": bool(settings.gemini_api_key),
                "openai": bool(settings.openai_api_key),
            },
        }

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile | None = File(default=None),
        local_path: str = Form(default=""),
        source_lang: str = Form(default="auto"),
        asr_backend: str = Form(default="auto"),
        whisper_model: str = Form(default="large-v3-turbo"),
        translator: str = Form(default="openai"),
        translation_model: str = Form(default=""),
        api_key: str = Form(default=""),
        api_base_url: str = Form(default=""),
        context_prompt: str = Form(default=""),
        batch_size: int = Form(default=30),
        bilingual: bool = Form(default=False),
        save_source_srt: bool = Form(default=False),
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]

        if file is not None and file.filename:
            job_dir = storage_root / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(file.filename).name
            dest = job_dir / safe_name
            with dest.open("wb") as out_f:
                shutil.copyfileobj(file.file, out_f)
            input_path = dest
            display_name = safe_name
            output_srt = job_dir / f"{dest.stem}{'.bilingual.srt' if bilingual else '.zh.srt'}"
        elif local_path.strip():
            resolved = Path(local_path.strip()).expanduser().resolve()
            if not resolved.exists() or not resolved.is_file():
                raise HTTPException(status_code=400, detail=f"Local file not found: {resolved}")
            input_path = resolved
            display_name = resolved.name
            output_srt = None  # Save beside the original local file.
        else:
            raise HTTPException(status_code=400, detail="Provide either an uploaded file or a local_path.")

        cfg = PipelineConfig(
            source_lang=source_lang,  # type: ignore[arg-type]
            asr_backend=asr_backend,  # type: ignore[arg-type]
            whisper_model=whisper_model.strip() or "large-v3-turbo",
            translator=translator,  # type: ignore[arg-type]
            translation_model=translation_model.strip() or None,
            api_key=api_key.strip() or None,
            api_base_url=api_base_url.strip() or None,
            context_prompt=context_prompt.strip(),
            batch_size=max(1, min(100, batch_size)),
            bilingual=bilingual,
            save_source_srt=save_source_srt,
        )

        record = JobRecord(
            job_id=job_id,
            filename=display_name,
            input_path=input_path,
            output_srt_path=output_srt,
        )
        jobs[job_id] = record

        def worker() -> None:
            record.status = "running"
            try:
                result = run_pipeline(
                    record.input_path,
                    output_path=record.output_srt_path,
                    config=cfg,
                    on_event=record.push_event,
                )
                with record.lock:
                    record.output_srt_path = Path(result.output_srt_path)
                    record.detected_language = result.detected_language
                    record.duration = result.duration
                    record.segments = result.segments
                    record.status = "completed"
            except Exception as exc:
                err_msg = str(exc)
                with record.lock:
                    record.status = "error"
                    record.error = err_msg
                record.push_event(
                    ProgressEvent(
                        stage="error",
                        progress=record.progress,
                        message=err_msg,
                    )
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        return {
            "job_id": job_id,
            "filename": display_name,
            "status": record.status,
        }

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return record.to_summary()

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: str) -> StreamingResponse:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        async def event_generator():
            sent = 0
            while True:
                with record.lock:
                    pending = list(record.events[sent:])
                    status = record.status
                for evt in pending:
                    sent += 1
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                if status in ("completed", "error") and sent >= len(record.events):
                    yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.put("/api/jobs/{job_id}/segments")
    def update_segments(job_id: str, payload: SegmentsUpdatePayload) -> dict[str, Any]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        updates = {item.index: item.translated_text for item in payload.segments}
        with record.lock:
            updated_list: list[SubtitleSegment] = []
            for seg in record.segments:
                if seg.index in updates:
                    updated_list.append(seg.model_copy(update={"translated_text": updates[seg.index]}))
                else:
                    updated_list.append(seg)
            record.segments = updated_list
            if record.output_srt_path is not None:
                write_srt(record.segments, record.output_srt_path, mode="translated")

        return {"updated": len(updates), "total": len(record.segments)}

    @app.get("/api/jobs/{job_id}/media")
    def get_job_media(job_id: str) -> FileResponse:
        record = jobs.get(job_id)
        if record is None or not record.is_media or not record.input_path.exists():
            raise HTTPException(status_code=404, detail="Media file not available for preview.")
        media_type, _ = mimetypes.guess_type(str(record.input_path))
        return FileResponse(path=str(record.input_path), media_type=media_type or "application/octet-stream")

    @app.get("/api/jobs/{job_id}/subtitles.vtt", response_class=PlainTextResponse)
    def get_job_vtt(
        job_id: str,
        mode: Literal["translated", "bilingual", "source"] = Query(default="translated"),
    ) -> PlainTextResponse:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        with record.lock:
            vtt = render_vtt(record.segments, mode=mode)
        return PlainTextResponse(content=vtt, media_type="text/vtt; charset=utf-8")

    @app.get("/api/jobs/{job_id}/download", response_class=PlainTextResponse)
    def download_job_srt(
        job_id: str,
        mode: Literal["translated", "bilingual", "source"] = Query(default="translated"),
    ) -> PlainTextResponse:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        with record.lock:
            srt_content = render_srt(record.segments, mode=mode)
            stem = Path(record.filename).stem
            lang_tag = (record.detected_language or "src").lower()

        suffix = {"translated": ".zh.srt", "bilingual": ".bilingual.srt", "source": f".{lang_tag}.srt"}[mode]
        download_name = f"{stem}{suffix}"
        encoded_name = quote(download_name)
        headers = {
            "Content-Disposition": f"attachment; filename=\"subtitles{suffix}\"; filename*=UTF-8''{encoded_name}"
        }
        return PlainTextResponse(content=srt_content, media_type="application/x-subrip; charset=utf-8", headers=headers)

    return app


app = create_app()
