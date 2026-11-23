"""Tests for cross-platform OS/GPU detection, automatic Whisper selection, and OpenVINO/CUDA backends."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wave

import numpy as np
import pytest

from subgen.config import resolve_faster_whisper_model, resolve_mlx_model_repo, resolve_openvino_model_repo
from subgen.hardware import GPUInfo, build_hardware_profile, resolve_asr_plan
from subgen.transcribe import FasterWhisperBackend, OpenVINOWhisperBackend


def _write_dummy_wav(path: Path, seconds: float = 0.5) -> Path:
    samples = np.zeros(int(16000 * seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return path


def test_macos_apple_silicon_profile():
    hw_16gb = build_hardware_profile(os_name="Darwin", arch="arm64", gpus=[], system_ram_mb=16384)
    assert hw_16gb.primary_accelerator == "apple_silicon"
    assert hw_16gb.recommended_backend == "mlx"
    assert hw_16gb.recommended_whisper_model == "large-v3-turbo"

    hw_8gb = build_hardware_profile(os_name="Darwin", arch="arm64", gpus=[], system_ram_mb=8192)
    assert hw_8gb.recommended_backend == "mlx"
    assert hw_8gb.recommended_whisper_model == "medium"


def test_windows_nvidia_dgpu_vram_tiers():
    # High VRAM (>= 6 GB) -> CUDA float16 + large-v3-turbo
    hw_4070 = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4070", vram_mb=12288, is_discrete=True)],
        system_ram_mb=32768,
    )
    assert hw_4070.primary_accelerator == "nvidia_cuda"
    assert hw_4070.recommended_backend == "faster-whisper"
    assert hw_4070.recommended_device == "cuda"
    assert hw_4070.recommended_compute_type == "float16"
    assert hw_4070.recommended_whisper_model == "large-v3-turbo"

    # Mid VRAM (4-6 GB) -> CUDA int8_float16 + large-v3-turbo
    hw_3050 = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="nvidia", name="NVIDIA GeForce RTX 3050 Laptop GPU", vram_mb=4096, is_discrete=True)],
        system_ram_mb=16384,
    )
    assert hw_3050.recommended_device == "cuda"
    assert hw_3050.recommended_compute_type == "int8_float16"
    assert hw_3050.recommended_whisper_model == "large-v3-turbo"

    # Low VRAM (< 4 GB) -> CUDA int8_float16 + medium
    hw_mx = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="nvidia", name="NVIDIA GeForce MX450", vram_mb=2048, is_discrete=True)],
        system_ram_mb=16384,
    )
    assert hw_mx.recommended_device == "cuda"
    assert hw_mx.recommended_compute_type == "int8_float16"
    assert hw_mx.recommended_whisper_model == "medium"


def test_windows_intel_igpu_profiles():
    # 16 GB RAM laptop with Intel Iris Xe -> OpenVINO GPU + int8 + large-v3-turbo
    hw_iris = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="intel", name="Intel(R) Iris(R) Xe Graphics", vram_mb=1024, is_discrete=False)],
        system_ram_mb=16384,
    )
    assert hw_iris.primary_accelerator == "intel_igpu"
    assert hw_iris.recommended_backend == "openvino"
    assert hw_iris.recommended_device == "GPU"
    assert hw_iris.recommended_compute_type == "int8"
    assert hw_iris.recommended_whisper_model == "large-v3-turbo"
    assert resolve_openvino_model_repo(hw_iris.recommended_whisper_model, hw_iris.recommended_compute_type) == (
        "OpenVINO/whisper-large-v3-turbo-int8-ov"
    )

    # 8 GB RAM laptop with Intel UHD -> OpenVINO GPU + int4 + large-v3-turbo-int4
    hw_uhd = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="intel", name="Intel(R) UHD Graphics", vram_mb=1024, is_discrete=False)],
        system_ram_mb=8192,
    )
    assert hw_uhd.recommended_backend == "openvino"
    assert hw_uhd.recommended_compute_type == "int4"
    assert hw_uhd.recommended_whisper_model == "large-v3-turbo-int4"
    assert resolve_openvino_model_repo(hw_uhd.recommended_whisper_model, hw_uhd.recommended_compute_type) == (
        "OpenVINO/whisper-large-v3-turbo-int4-ov"
    )


def test_windows_hybrid_laptop_prefers_nvidia_and_allows_intel_override():
    hw_hybrid = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[
            GPUInfo(vendor="intel", name="Intel(R) Iris(R) Xe Graphics", vram_mb=1024, is_discrete=False),
            GPUInfo(vendor="nvidia", name="NVIDIA GeForce RTX 3060 Laptop GPU", vram_mb=6144, is_discrete=True),
        ],
        system_ram_mb=16384,
    )
    installed = {"mlx": False, "faster-whisper": True, "openvino": True}

    auto_plan = resolve_asr_plan("auto", "auto", hardware=hw_hybrid, installed=installed)
    assert auto_plan.backend == "faster-whisper"
    assert auto_plan.device == "cuda"
    assert auto_plan.compute_type == "float16"
    assert auto_plan.model_name == "large-v3-turbo"

    intel_plan = resolve_asr_plan("openvino", "auto", hardware=hw_hybrid, installed=installed)
    assert intel_plan.backend == "openvino"
    assert intel_plan.device == "GPU"
    assert intel_plan.model_name == "large-v3-turbo"


def test_intel_igpu_falls_back_to_cpu_when_openvino_not_installed():
    hw_intel = build_hardware_profile(
        os_name="Windows",
        arch="AMD64",
        gpus=[GPUInfo(vendor="intel", name="Intel(R) Iris(R) Xe Graphics", is_discrete=False)],
        system_ram_mb=16384,
    )
    plan = resolve_asr_plan(
        "auto",
        "auto",
        hardware=hw_intel,
        installed={"mlx": False, "faster-whisper": True, "openvino": False},
    )
    assert plan.backend == "faster-whisper"
    assert plan.device == "cpu"
    assert plan.compute_type == "int8"
    assert plan.model_name == "medium"


def test_model_repo_resolution_helpers():
    assert resolve_mlx_model_repo("large-v3-turbo-int4") == "mlx-community/whisper-large-v3-turbo"
    assert resolve_faster_whisper_model("large-v3-turbo-int4") == "large-v3-turbo"
    assert resolve_openvino_model_repo("large-v3-turbo", "int4") == "OpenVINO/whisper-large-v3-turbo-int4-ov"
    assert resolve_openvino_model_repo("custom/my-ov-whisper", "int8") == "custom/my-ov-whisper"


def test_openvino_backend_transcribe_and_gpu_fallback(tmp_path: Path):
    wav = _write_dummy_wav(tmp_path / "sample.wav")
    model_dir = tmp_path / "ov_model"
    model_dir.mkdir()

    created_devices: list[str] = []

    class FakePipeline:
        def __init__(self, path, device, **kwargs):
            created_devices.append(device)
            if device == "GPU":
                raise RuntimeError("Intel GPU OpenCL driver missing")

        def generate(self, speech, **kwargs):
            assert kwargs["return_timestamps"] is True
            return SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        start_ts=0.0,
                        end_ts=1.5,
                        text="おはようございます。",
                        words=[SimpleNamespace(word="おはようございます。", start_ts=0.0, end_ts=1.5)],
                    )
                ]
            )

    fake_ov_genai = SimpleNamespace(WhisperPipeline=FakePipeline)
    backend = OpenVINOWhisperBackend(device="GPU", compute_type="int8")

    with patch.dict("sys.modules", {"openvino_genai": fake_ov_genai}):
        result = backend.transcribe(wav, source_lang="auto", model_name=str(model_dir))

    assert created_devices == ["GPU", "CPU"]
    assert result.language == "ja"
    assert len(result.segments) == 1
    assert result.segments[0].source_text == "おはようございます。"


def test_faster_whisper_backend_cuda_falls_back_to_cpu(tmp_path: Path):
    wav = _write_dummy_wav(tmp_path / "sample.wav")
    init_calls: list[tuple[str, str, str]] = []

    class FakeWhisperModel:
        def __init__(self, model_id, device="auto", compute_type="default"):
            init_calls.append((model_id, device, compute_type))
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")

        def transcribe(self, audio, **kwargs):
            seg = SimpleNamespace(start=0.0, end=1.0, text="Hello world", words=[])
            info = SimpleNamespace(language="en", duration=1.0)
            return iter([seg]), info

    fake_fw = SimpleNamespace(WhisperModel=FakeWhisperModel)
    backend = FasterWhisperBackend(device="cuda", compute_type="float16")

    with patch.dict("sys.modules", {"faster_whisper": fake_fw}):
        result = backend.transcribe(wav, source_lang="en", model_name="large-v3-turbo")

    assert init_calls == [("large-v3-turbo", "cuda", "float16"), ("large-v3-turbo", "cpu", "int8")]
    assert result.language == "en"
    assert result.segments[0].source_text == "Hello world"
