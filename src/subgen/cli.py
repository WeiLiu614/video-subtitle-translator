"""Command-line interface for SubGen."""

from __future__ import annotations

import platform
import sys
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from subgen import __version__
from subgen.audio import FFmpegNotFoundError, find_ffmpeg
from subgen.config import load_settings
from subgen.hardware import detect_hardware, has_module
from subgen.models import PipelineConfig, ProgressEvent
from subgen.pipeline import run_pipeline
from subgen.srt import format_srt_timestamp

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="subgen")
def main() -> None:
    """SubGen — Generate Simplified Chinese (.srt) subtitles for Japanese & English videos."""


@main.command("generate")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None, help="Output .srt file path (default: <name>.zh.srt).")
@click.option("-l", "--lang", "source_lang", type=click.Choice(["auto", "ja", "en"]), default="auto", show_default=True, help="Source audio language.")
@click.option("--asr", "asr_backend", type=click.Choice(["auto", "mlx", "faster-whisper", "openvino"]), default=None, help="Local Whisper backend (default: auto-detected from OS & GPU).")
@click.option("--whisper-model", default=None, help="Whisper model alias, Hugging Face repo, or 'auto' (hardware-selected default).")
@click.option("-t", "--translator", type=click.Choice(["gemini", "openai"]), default=None, help="Cloud LLM provider for translation.")
@click.option("-m", "--model", "translation_model", default=None, help="Translation LLM model name override.")
@click.option("--api-key", default=None, help="LLM API key override (otherwise read from .env / environment).")
@click.option("--base-url", "api_base_url", default=None, help="Custom base URL for OpenAI-compatible providers (DeepSeek, OpenRouter, Ollama, etc.).")
@click.option("--batch-size", default=30, show_default=True, type=click.IntRange(1, 100), help="Subtitle cues translated per LLM batch.")
@click.option("--context", "context_prompt", default="", help="Optional video topic, tone, or proper-noun glossary to guide translation.")
@click.option("--save-source", is_flag=True, default=False, help="Also save the original transcribed Japanese/English .srt file.")
@click.option("--bilingual", is_flag=True, default=False, help="Include both Simplified Chinese and original text in the output .srt.")
def generate_cmd(
    input_path: Path,
    output_path: Path | None,
    source_lang: str,
    asr_backend: str | None,
    whisper_model: str | None,
    translator: str | None,
    translation_model: str | None,
    api_key: str | None,
    api_base_url: str | None,
    batch_size: int,
    context_prompt: str,
    save_source: bool,
    bilingual: bool,
) -> None:
    """Generate Simplified Chinese .srt subtitles from a video, audio, or source .srt file."""
    settings = load_settings()

    cfg = PipelineConfig(
        source_lang=source_lang,  # type: ignore[arg-type]
        asr_backend=(asr_backend or settings.default_asr_backend),  # type: ignore[arg-type]
        whisper_model=whisper_model or settings.default_whisper_model,
        translator=(translator or settings.default_translator),  # type: ignore[arg-type]
        translation_model=translation_model,
        api_key=api_key,
        api_base_url=api_base_url,
        batch_size=batch_size,
        context_prompt=context_prompt,
        save_source_srt=save_source,
        bilingual=bilingual,
    )

    console.print(
        Panel.fit(
            f"[bold cyan]SubGen[/bold cyan]  [dim]v{__version__}[/dim]\n"
            f"Input: [white]{input_path.name}[/white]  |  Lang: [yellow]{cfg.source_lang}[/yellow]  |  "
            f"ASR: [green]{cfg.asr_backend} ({cfg.whisper_model})[/green]  |  "
            f"LLM: [magenta]{cfg.translator}[/magenta]",
            border_style="cyan",
        )
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Starting...", total=100.0)

            def handle_event(event: ProgressEvent) -> None:
                progress.update(
                    task_id,
                    completed=event.progress,
                    description=f"[cyan]{event.stage.capitalize()}[/cyan]: {event.message}",
                )

            result = run_pipeline(
                input_path,
                output_path=output_path,
                config=cfg,
                on_event=handle_event,
            )
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"\n[bold green]✓ Saved Chinese subtitles:[/bold green] {result.output_srt_path}")
    if result.source_srt_path:
        console.print(f"[dim]  Source subtitles saved:[/dim] {result.source_srt_path}")

    preview_count = min(5, len(result.segments))
    if preview_count > 0:
        table = Table(title=f"Subtitle Preview (first {preview_count} of {len(result.segments)} cues)", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Time", style="cyan", width=25)
        table.add_column(f"Source ({result.detected_language})", style="white")
        table.add_column("简体中文", style="bold green")

        for seg in result.segments[:preview_count]:
            ts = f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}"
            table.add_row(str(seg.index), ts, seg.source_text, seg.translated_text)
        console.print()
        console.print(table)


@main.command("web")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host address to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port number to serve the Web UI.")
@click.option("--no-open", is_flag=True, default=False, help="Do not automatically open the browser.")
def web_cmd(host: str, port: int, no_open: bool) -> None:
    """Launch the local SubGen Web UI (drag-and-drop, live preview & subtitle editor)."""
    import uvicorn

    url = f"http://{host}:{port}"
    console.print(
        Panel.fit(
            f"[bold cyan]SubGen Web UI[/bold cyan]\n"
            f"Open in browser: [link={url}]{url}[/link]\n"
            f"[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )
    if not no_open:
        webbrowser.open(url)

    uvicorn.run("subgen.web.app:app", host=host, port=port, reload=False, log_level="info")


@main.command("doctor")
def doctor_cmd() -> None:
    """Check OS, GPU/hardware detection, Whisper backends, FFmpeg, and LLM API configuration."""
    settings = load_settings()
    hw = detect_hardware()

    table = Table(title="SubGen Environment Diagnostics")
    table.add_column("Component", style="cyan", width=24)
    table.add_column("Status", width=14)
    table.add_column("Details")

    ram_info = f", {hw.system_ram_mb // 1024} GB RAM" if hw.system_ram_mb else ""
    table.add_row(
        "Platform",
        "[green]OK[/green]",
        f"{hw.os_name} {hw.arch} ({hw.primary_accelerator}{ram_info})",
    )

    if hw.gpus:
        gpu_desc = "; ".join(
            f"{g.name}" + (f" ({g.vram_mb} MB)" if g.vram_mb else "")
            for g in hw.gpus
        )
    else:
        gpu_desc = "No supported GPU detected (CPU mode)"
    table.add_row("Detected GPU(s)", "[green]OK[/green]" if hw.gpus else "[dim]CPU[/dim]", gpu_desc)
    table.add_row(
        "Auto Whisper Plan",
        "[cyan]AUTO[/cyan]",
        f"{hw.recommended_backend} | device={hw.recommended_device} | precision={hw.recommended_compute_type} | model={hw.recommended_whisper_model}",
    )
    table.add_row("Python", "[green]OK[/green]", sys.version.split()[0])

    try:
        ffmpeg_bin = find_ffmpeg()
        table.add_row("FFmpeg", "[green]OK[/green]", ffmpeg_bin)
    except FFmpegNotFoundError as exc:
        table.add_row("FFmpeg", "[red]MISSING[/red]", str(exc))

    mlx_ok = has_module("mlx_whisper")
    table.add_row(
        "mlx-whisper",
        "[green]INSTALLED[/green]" if mlx_ok else ("[yellow]RECOMMENDED[/yellow]" if hw.recommended_backend == "mlx" else "[dim]OPTIONAL[/dim]"),
        "Apple Silicon Metal GPU backend" if mlx_ok else r"macOS Apple Silicon: pip install -e '.\[mlx]'",
    )

    fw_ok = has_module("faster_whisper")
    table.add_row(
        "faster-whisper",
        "[green]INSTALLED[/green]" if fw_ok else ("[yellow]RECOMMENDED[/yellow]" if hw.recommended_backend == "faster-whisper" else "[dim]OPTIONAL[/dim]"),
        "NVIDIA CUDA dGPU & CPU backend (CTranslate2 + VAD)" if fw_ok else r"Windows/Linux NVIDIA or CPU: pip install -e '.\[nvidia]'",
    )

    ov_ok = has_module("openvino_genai")
    table.add_row(
        "openvino-genai",
        "[green]INSTALLED[/green]" if ov_ok else ("[yellow]RECOMMENDED[/yellow]" if hw.recommended_backend == "openvino" else "[dim]OPTIONAL[/dim]"),
        "Intel iGPU / Arc GPU backend (OpenVINO)" if ov_ok else r"Windows/Linux Intel iGPU: pip install -e '.\[intel]'",
    )

    table.add_row(
        "OPENAI_API_KEY",
        "[green]CONFIGURED[/green]" if settings.openai_api_key else "[yellow]NOT SET[/yellow]",
        f"Default model: {settings.default_openai_model}"
        + (f" | Base URL: {settings.openai_base_url}" if settings.openai_base_url else ""),
    )
    table.add_row(
        "GEMINI_API_KEY",
        "[green]CONFIGURED[/green]" if settings.gemini_api_key else "[dim]NOT SET[/dim]",
        f"Default model: {settings.default_gemini_model}",
    )

    console.print(table)


if __name__ == "__main__":
    main()
