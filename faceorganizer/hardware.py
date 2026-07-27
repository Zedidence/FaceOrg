"""Hardware probe — detect GPU, CPU, RAM and derive optimal runtime settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeProfile:
    """Hardware capabilities and derived runtime recommendations."""

    gpu_provider: str                   # "CUDAExecutionProvider" | "DmlExecutionProvider" | "CPUExecutionProvider"
    physical_cores: int
    logical_cores: int
    ram_gb: float
    recommended_workers: int
    use_gpu: bool
    thumbnail_resolution: tuple[int, int]
    provider_list: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable hardware summary for the Settings panel."""
        gpu_label = (
            "NVIDIA CUDA GPU"
            if self.gpu_provider == "CUDAExecutionProvider"
            else "DirectML GPU"
            if self.gpu_provider == "DmlExecutionProvider"
            else "No GPU (CPU only)"
        )
        return (
            f"{self.physical_cores} physical cores ({self.logical_cores} logical)  •  "
            f"{self.ram_gb:.1f} GB RAM  •  {gpu_label}  •  "
            f"Recommended workers: {self.recommended_workers}"
        )


def probe_hardware() -> RuntimeProfile:
    """Detect system hardware and return a RuntimeProfile with recommendations."""
    import onnxruntime as ort
    import psutil

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        gpu = "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in available:
        gpu = "DmlExecutionProvider"
    else:
        gpu = "CPUExecutionProvider"
    use_gpu = gpu != "CPUExecutionProvider"

    provider_list = [gpu, "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    phys = psutil.cpu_count(logical=False) or 1
    logi = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Leave one physical core free for the UI thread
    if use_gpu:
        workers = min(4, max(1, phys - 1))
    else:
        workers = max(1, phys - 1)

    # Higher-res thumbnails on machines with plenty of RAM
    thumb_res = (200, 200) if ram_gb >= 16 else (150, 150)

    return RuntimeProfile(
        gpu_provider=gpu,
        physical_cores=phys,
        logical_cores=logi,
        ram_gb=ram_gb,
        recommended_workers=workers,
        use_gpu=use_gpu,
        thumbnail_resolution=thumb_res,
        provider_list=provider_list,
    )
