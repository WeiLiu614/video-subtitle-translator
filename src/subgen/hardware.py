"""Cross-platform OS and GPU/accelerator detection for automatic Whisper selection."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, Field

AcceleratorType = Literal["apple_silicon", "nvidia_cuda", "intel_igpu", "cpu"]
GPUVendor = Literal["apple", "nvidia", "intel", "amd", "unknown"]
ConcreteASRBackend = Literal["mlx", "faster-whisper", "openvino"]


class GPUInfo(BaseModel):
    """Information about a detected GPU/accelerator."""

    vendor: GPUVendor
    name: str
    vram_mb: int | None = None
    is_discrete: bool = False


class HardwareProfile(BaseModel):
    """Detected host OS, GPUs, and recommended Whisper execution profile."""

    os_name: str
    arch: str
    is_windows: bool = False
    is_macos: bool = False
    is_apple_silicon: bool = False
    gpus: list[GPUInfo] = Field(default_factory=list)
    primary_accelerator: AcceleratorType = "cpu"
    primary_gpu_name: str | None = None
    vram_mb: int | None = None
    system_ram_mb: int | None = None
    recommended_backend: ConcreteASRBackend = "faster-whisper"
    recommended_device: str = "cpu"
    recommended_compute_type: str = "int8"
    recommended_whisper_model: str = "medium"
    reason: str = ""


@dataclass(frozen=True)
class ASRPlan:
    """Concrete resolved backend, device, compute type, and model for a transcription run."""

    backend: ConcreteASRBackend
    device: str
    compute_type: str
    model_name: str
    hardware: HardwareProfile


def has_module(module_name: str) -> bool:
    """Return True if a Python module is importable without importing it."""
    return importlib.util.find_spec(module_name) is not None


def _classify_vendor(name: str) -> GPUVendor:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low or "gtx" in low or "tesla" in low:
        return "nvidia"
    if "intel" in low or "iris" in low or "uhd graphics" in low or "hd graphics" in low or "arc" in low:
        return "intel"
    if "apple" in low:
        return "apple"
    if "amd" in low or "radeon" in low:
        return "amd"
    return "unknown"


def _is_discrete_gpu(vendor: GPUVendor, name: str) -> bool:
    low = name.lower()
    if vendor == "nvidia":
        return True
    if vendor == "intel":
        # Intel Arc A-series / B-series dGPUs (e.g. "Intel Arc A770") are discrete;
        # "Arc Graphics" (Meteor Lake / Lunar Lake) / Iris Xe / UHD are integrated.
        return "arc a" in low or "arc b" in low or "arc pro" in low
    if vendor == "amd":
        return "radeon(tm) graphics" not in low
    return False


def _detect_system_ram_mb() -> int | None:
    """Best-effort total system RAM detection in MiB across macOS, Windows, and Linux."""
    system = platform.system()
    try:
        if system == "Windows":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys // (1024 * 1024))
        elif system == "Darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                return int(proc.stdout.strip()) // (1024 * 1024)
        elif system == "Linux" and os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
    except Exception:
        return None
    return None


def _probe_nvidia_smi() -> list[GPUInfo]:
    """Query NVIDIA GPUs and VRAM via `nvidia-smi` (works on Windows & Linux)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            check=False,
        )
        if proc.returncode != 0:
            return []
        gpus: list[GPUInfo] = []
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            vram_mb: int | None = None
            if len(parts) >= 2:
                try:
                    vram_mb = int(float(parts[1]))
                except ValueError:
                    vram_mb = None
            gpus.append(GPUInfo(vendor="nvidia", name=name, vram_mb=vram_mb, is_discrete=True))
        return gpus
    except Exception:
        return []


def _probe_windows_gpus() -> list[GPUInfo]:
    """Query Windows display adapters via PowerShell CIM (covers Intel iGPU, NVIDIA, AMD)."""
    if platform.system() != "Windows":
        return []
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return []
    cmd = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json -Compress",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        raw = json.loads(proc.stdout.strip())
        items = raw if isinstance(raw, list) else [raw]
        gpus: list[GPUInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name") or "").strip()
            if not name:
                continue
            vendor = _classify_vendor(name)
            vram_mb: int | None = None
            adapter_ram = item.get("AdapterRAM")
            if isinstance(adapter_ram, (int, float)) and adapter_ram > 0:
                vram_mb = int(adapter_ram // (1024 * 1024))
            gpus.append(
                GPUInfo(
                    vendor=vendor,
                    name=name,
                    vram_mb=vram_mb,
                    is_discrete=_is_discrete_gpu(vendor, name),
                )
            )
        return gpus
    except Exception:
        return []


def _probe_openvino_gpus() -> list[GPUInfo]:
    """Query Intel GPUs visible to OpenVINO runtime if installed."""
    if not has_module("openvino"):
        return []
    try:
        import openvino as ov  # type: ignore[import-untyped]

        core = ov.Core()
        gpus: list[GPUInfo] = []
        for dev in core.available_devices:
            if dev == "GPU" or dev.startswith("GPU."):
                try:
                    full_name = str(core.get_property(dev, "FULL_DEVICE_NAME"))
                except Exception:
                    full_name = f"Intel OpenVINO {dev}"
                vendor = _classify_vendor(full_name)
                if vendor == "unknown":
                    vendor = "intel"
                gpus.append(
                    GPUInfo(
                        vendor=vendor,
                        name=full_name,
                        vram_mb=None,
                        is_discrete=_is_discrete_gpu(vendor, full_name),
                    )
                )
        return gpus
    except Exception:
        return []


def _probe_cuda_python() -> list[GPUInfo]:
    """Check CUDA availability via PyTorch or CTranslate2 as a fallback to nvidia-smi."""
    if has_module("torch"):
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                gpus: list[GPUInfo] = []
                for idx in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(idx)
                    gpus.append(
                        GPUInfo(
                            vendor="nvidia",
                            name=str(props.name),
                            vram_mb=int(props.total_memory // (1024 * 1024)),
                            is_discrete=True,
                        )
                    )
                return gpus
        except Exception:
            pass

    if has_module("ctranslate2"):
        try:
            import ctranslate2  # type: ignore[import-untyped]

            if ctranslate2.get_cuda_device_count() > 0:
                return [GPUInfo(vendor="nvidia", name="NVIDIA CUDA GPU", vram_mb=None, is_discrete=True)]
        except Exception:
            pass

    return []


def _probe_linux_intel_gpu() -> list[GPUInfo]:
    """Detect Intel GPU on Linux via PCI vendor ID (`0x8086`) in sysfs."""
    if platform.system() != "Linux":
        return []
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return []
    try:
        for entry in sorted(os.listdir(drm_root)):
            vendor_file = os.path.join(drm_root, entry, "device", "vendor")
            if os.path.isfile(vendor_file):
                with open(vendor_file, encoding="utf-8") as fh:
                    if fh.read().strip().lower() == "0x8086":
                        return [GPUInfo(vendor="intel", name="Intel Integrated Graphics", vram_mb=None, is_discrete=False)]
    except Exception:
        return []
    return []


def _merge_gpus(*groups: Sequence[GPUInfo]) -> list[GPUInfo]:
    """Deduplicate GPU entries while preserving the richest VRAM info (prefer nvidia-smi over WMI)."""
    merged: list[GPUInfo] = []
    for group in groups:
        for gpu in group:
            key = gpu.name.lower().strip()
            existing_idx = next(
                (i for i, item in enumerate(merged) if item.name.lower().strip() == key),
                None,
            )
            if existing_idx is None:
                # Avoid duplicating generic "NVIDIA CUDA GPU" when a named NVIDIA GPU already exists.
                if gpu.vendor == "nvidia" and key == "nvidia cuda gpu" and any(g.vendor == "nvidia" for g in merged):
                    continue
                merged.append(gpu)
            else:
                prev = merged[existing_idx]
                # Prefer accurate VRAM (WMI AdapterRAM caps at 4095 MB on 32-bit unsigned int).
                best_vram = prev.vram_mb
                if gpu.vram_mb is not None and (best_vram is None or gpu.vram_mb > best_vram):
                    best_vram = gpu.vram_mb
                merged[existing_idx] = prev.model_copy(update={"vram_mb": best_vram})
    return merged


def build_hardware_profile(
    *,
    os_name: str,
    arch: str,
    gpus: Sequence[GPUInfo],
    system_ram_mb: int | None = None,
) -> HardwareProfile:
    """Compute recommended ASR backend, device, precision, and Whisper model from OS & GPU info."""
    is_macos = os_name == "Darwin"
    is_windows = os_name == "Windows"
    is_apple_silicon = is_macos and arch.lower() in ("arm64", "aarch64")

    gpu_list = list(gpus)
    if is_apple_silicon and not any(g.vendor == "apple" for g in gpu_list):
        gpu_list.insert(0, GPUInfo(vendor="apple", name="Apple Silicon GPU (Metal)", vram_mb=system_ram_mb, is_discrete=False))

    # Priority 1: Apple Silicon (macOS arm64) -> mlx-whisper
    if is_apple_silicon:
        # On 8 GB Macs, medium is safer; 16 GB+ handles large-v3-turbo effortlessly.
        model = "medium" if (system_ram_mb is not None and system_ram_mb <= 8192) else "large-v3-turbo"
        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            is_windows=False,
            is_macos=True,
            is_apple_silicon=True,
            gpus=gpu_list,
            primary_accelerator="apple_silicon",
            primary_gpu_name=gpu_list[0].name,
            vram_mb=system_ram_mb,
            system_ram_mb=system_ram_mb,
            recommended_backend="mlx",
            recommended_device="mps",
            recommended_compute_type="default",
            recommended_whisper_model=model,
            reason=f"macOS Apple Silicon detected -> MLX GPU acceleration with `{model}`",
        )

    # Priority 2: NVIDIA dGPU (Windows / Linux) -> faster-whisper (CUDA)
    nvidia_gpus = [g for g in gpu_list if g.vendor == "nvidia"]
    if nvidia_gpus:
        best_nv = max(nvidia_gpus, key=lambda g: g.vram_mb or 0)
        vram = best_nv.vram_mb
        if vram is not None and vram < 4000:
            compute_type = "int8_float16"
            model = "medium"
            tier_note = f"{vram} MB VRAM (<4 GB) -> `medium` ({compute_type})"
        elif vram is not None and vram < 6000:
            compute_type = "int8_float16"
            model = "large-v3-turbo"
            tier_note = f"{vram} MB VRAM (4-6 GB) -> `large-v3-turbo` ({compute_type})"
        else:
            compute_type = "float16"
            model = "large-v3-turbo"
            vram_label = f"{vram} MB VRAM" if vram else "CUDA"
            tier_note = f"{vram_label} -> `large-v3-turbo` ({compute_type})"

        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            is_windows=is_windows,
            is_macos=is_macos,
            is_apple_silicon=False,
            gpus=gpu_list,
            primary_accelerator="nvidia_cuda",
            primary_gpu_name=best_nv.name,
            vram_mb=vram,
            system_ram_mb=system_ram_mb,
            recommended_backend="faster-whisper",
            recommended_device="cuda",
            recommended_compute_type=compute_type,
            recommended_whisper_model=model,
            reason=f"{os_name} + NVIDIA dGPU ({best_nv.name}, {tier_note})",
        )

    # Priority 3: Intel iGPU / Arc (Windows / Linux) -> OpenVINO GPU
    intel_gpus = [g for g in gpu_list if g.vendor == "intel"]
    if intel_gpus:
        best_intel = intel_gpus[0]
        # On 8 GB RAM laptops with Intel iGPU, INT4 turbo uses much less shared memory.
        if system_ram_mb is not None and system_ram_mb <= 10240:
            compute_type = "int4"
            model = "large-v3-turbo-int4"
        else:
            compute_type = "int8"
            model = "large-v3-turbo"

        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            is_windows=is_windows,
            is_macos=is_macos,
            is_apple_silicon=False,
            gpus=gpu_list,
            primary_accelerator="intel_igpu",
            primary_gpu_name=best_intel.name,
            vram_mb=best_intel.vram_mb,
            system_ram_mb=system_ram_mb,
            recommended_backend="openvino",
            recommended_device="GPU",
            recommended_compute_type=compute_type,
            recommended_whisper_model=model,
            reason=f"{os_name} + Intel GPU ({best_intel.name}) -> OpenVINO GPU with `{model}` ({compute_type})",
        )

    # Priority 4: CPU fallback (Intel Mac, Linux/Windows CPU-only)
    return HardwareProfile(
        os_name=os_name,
        arch=arch,
        is_windows=is_windows,
        is_macos=is_macos,
        is_apple_silicon=False,
        gpus=gpu_list,
        primary_accelerator="cpu",
        primary_gpu_name=gpu_list[0].name if gpu_list else None,
        vram_mb=gpu_list[0].vram_mb if gpu_list else None,
        system_ram_mb=system_ram_mb,
        recommended_backend="faster-whisper",
        recommended_device="cpu",
        recommended_compute_type="int8",
        recommended_whisper_model="medium",
        reason=f"{os_name} CPU fallback -> faster-whisper CPU (`int8`) with `medium`",
    )


def detect_hardware() -> HardwareProfile:
    """Detect current OS, GPUs, RAM, and optimal Whisper configuration."""
    os_name = platform.system()
    arch = platform.machine()
    ram_mb = _detect_system_ram_mb()

    if os_name == "Darwin" and arch.lower() in ("arm64", "aarch64"):
        return build_hardware_profile(os_name=os_name, arch=arch, gpus=[], system_ram_mb=ram_mb)

    nvidia_smi_gpus = _probe_nvidia_smi()
    cuda_py_gpus = _probe_cuda_python() if not nvidia_smi_gpus else []
    win_gpus = _probe_windows_gpus()
    ov_gpus = _probe_openvino_gpus()
    linux_intel = _probe_linux_intel_gpu()

    # Order matters: nvidia-smi & PyTorch report true VRAM (>4 GB), whereas Windows WMI caps AdapterRAM at 4 GB.
    merged = _merge_gpus(nvidia_smi_gpus, cuda_py_gpus, ov_gpus, win_gpus, linux_intel)
    return build_hardware_profile(os_name=os_name, arch=arch, gpus=merged, system_ram_mb=ram_mb)


def resolve_asr_plan(
    requested_backend: str = "auto",
    requested_model: str = "auto",
    *,
    hardware: HardwareProfile | None = None,
    installed: dict[str, bool] | None = None,
) -> ASRPlan:
    """Resolve user-requested backend/model + detected hardware + installed packages into an ASRPlan."""
    hw = hardware or detect_hardware()
    avail = installed if installed is not None else {
        "mlx": has_module("mlx_whisper"),
        "faster-whisper": has_module("faster_whisper"),
        "openvino": has_module("openvino_genai"),
    }

    def _default_model_for(backend: ConcreteASRBackend, device: str) -> str:
        if requested_model and requested_model.strip().lower() != "auto":
            return requested_model.strip()
        if backend == hw.recommended_backend and device == hw.recommended_device:
            return hw.recommended_whisper_model
        if backend == "mlx":
            return "large-v3-turbo"
        if backend == "openvino":
            return "large-v3-turbo-int4" if (hw.system_ram_mb and hw.system_ram_mb <= 10240) else "large-v3-turbo"
        # faster-whisper fallback on CPU
        if device == "cuda":
            return hw.recommended_whisper_model
        return "medium"

    if requested_backend == "mlx":
        if not avail.get("mlx"):
            raise RuntimeError("mlx-whisper is not installed. Install with: pip install -e '.[mlx]'")
        return ASRPlan(
            backend="mlx",
            device="mps",
            compute_type="default",
            model_name=_default_model_for("mlx", "mps"),
            hardware=hw,
        )

    if requested_backend == "openvino":
        if not avail.get("openvino"):
            raise RuntimeError(
                "openvino-genai is not installed. Install for Intel iGPU with: pip install -e '.[intel]'"
            )
        device = "GPU" if hw.primary_accelerator == "intel_igpu" or any(g.vendor == "intel" for g in hw.gpus) else "CPU"
        return ASRPlan(
            backend="openvino",
            device=device,
            compute_type=hw.recommended_compute_type if hw.recommended_backend == "openvino" else "int8",
            model_name=_default_model_for("openvino", device),
            hardware=hw,
        )

    if requested_backend == "faster-whisper":
        if not avail.get("faster-whisper"):
            raise RuntimeError(
                "faster-whisper is not installed. Install for NVIDIA/CPU with: pip install -e '.[nvidia]'"
            )
        if hw.primary_accelerator == "nvidia_cuda":
            device = "cuda"
            compute_type = hw.recommended_compute_type
        else:
            device = "cpu"
            compute_type = "int8"
        return ASRPlan(
            backend="faster-whisper",
            device=device,
            compute_type=compute_type,
            model_name=_default_model_for("faster-whisper", device),
            hardware=hw,
        )

    # Auto mode: try hardware-recommended backend first, then gracefully fall back to any installed backend.
    if hw.recommended_backend == "mlx" and avail.get("mlx"):
        return ASRPlan(
            backend="mlx",
            device=hw.recommended_device,
            compute_type=hw.recommended_compute_type,
            model_name=_default_model_for("mlx", hw.recommended_device),
            hardware=hw,
        )

    if hw.recommended_backend == "faster-whisper" and avail.get("faster-whisper"):
        return ASRPlan(
            backend="faster-whisper",
            device=hw.recommended_device,
            compute_type=hw.recommended_compute_type,
            model_name=_default_model_for("faster-whisper", hw.recommended_device),
            hardware=hw,
        )

    if hw.recommended_backend == "openvino" and avail.get("openvino"):
        return ASRPlan(
            backend="openvino",
            device=hw.recommended_device,
            compute_type=hw.recommended_compute_type,
            model_name=_default_model_for("openvino", hw.recommended_device),
            hardware=hw,
        )

    # Graceful fallbacks if another local Whisper backend is installed
    if avail.get("faster-whisper"):
        device = "cuda" if hw.primary_accelerator == "nvidia_cuda" else "cpu"
        compute_type = hw.recommended_compute_type if device == "cuda" else "int8"
        return ASRPlan(
            backend="faster-whisper",
            device=device,
            compute_type=compute_type,
            model_name=_default_model_for("faster-whisper", device),
            hardware=hw,
        )

    if avail.get("openvino"):
        device = "GPU" if any(g.vendor == "intel" for g in hw.gpus) else "CPU"
        return ASRPlan(
            backend="openvino",
            device=device,
            compute_type="int8",
            model_name=_default_model_for("openvino", device),
            hardware=hw,
        )

    if avail.get("mlx"):
        return ASRPlan(
            backend="mlx",
            device="mps",
            compute_type="default",
            model_name=_default_model_for("mlx", "mps"),
            hardware=hw,
        )

    install_hint = {
        "apple_silicon": "pip install -e '.[mlx]'",
        "nvidia_cuda": "pip install -e '.[nvidia]'",
        "intel_igpu": "pip install -e '.[intel]'",
        "cpu": "pip install -e '.[faster-whisper]'",
    }[hw.primary_accelerator]
    raise RuntimeError(
        f"No local Whisper backend is installed for {hw.reason}. Install the recommended extra: {install_hint}"
    )
