from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HardwareWarningLevel(str, Enum):
    OK = "ok"
    WARNING = "warning"      
    CRITICAL = "critical"     

@dataclass
class HardwareReport:
    level: HardwareWarningLevel
    message: str
    available_ram_gb: float
    available_vram_gb: float | None
    estimated_model_bytes: int
    estimated_peak_bytes: int 
    device_used: str            # "cpu" | "cuda:0" | "mps" | ...


class HardwareCapabilityChecker:
    """
    Estimates whether a model will fit into available memory BEFORE
    loading/training it. Not tied to any specific ML framework — accepts
    an already-computed model size (from FrameworkAdapter.model_size_bytes)
    plus training/inference parameters.
    """
    INFERENCE_OVERHEAD_MULTIPLIER = 1.2
    TRAINING_OVERHEAD_MULTIPLIER = 4.0

    def __init__(self, safety_margin: float = 0.85):
        self._safety_margin = safety_margin

    def get_system_resources(self) -> dict[str, Any]:
        import psutil

        vmem = psutil.virtual_memory()
        resources: dict[str, Any] = {
            "total_ram_gb": vmem.total / (1024 ** 3),
            "available_ram_gb": vmem.available / (1024 ** 3),
            "cpu_count": psutil.cpu_count(logical=True),
            "gpus": [],
        }

        gpu_info = self._get_gpu_info()
        resources["gpus"] = gpu_info

        return resources

    def check_model_fit(
        self,
        model_size_bytes: int,
        mode: str = "inference",  # "inference" | "training"
        prefer_gpu: bool = True,
    ) -> HardwareReport:
        resources = self.get_system_resources()
        multiplier = (
            self.TRAINING_OVERHEAD_MULTIPLIER if mode == "training"
            else self.INFERENCE_OVERHEAD_MULTIPLIER
        )
        estimated_peak = int(model_size_bytes * multiplier)

        device, available_bytes = self._select_device(resources, prefer_gpu)
        available_gb = available_bytes / (1024 ** 3)
        usable_bytes = available_bytes * self._safety_margin

        vram_gb = None
        for gpu in resources["gpus"]:
            if f"cuda:{gpu['index']}" == device:
                vram_gb = gpu["free_vram_gb"]

        if estimated_peak > available_bytes:
            level = HardwareWarningLevel.CRITICAL
            message = (
                f"Model likely will NOT fit on {device}: estimated peak "
                f"{estimated_peak / (1024**3):.2f} GB > available {available_gb:.2f} GB. "
                f"Consider quantization, CPU offload, or a smaller model."
            )
        elif estimated_peak > usable_bytes:
            level = HardwareWarningLevel.WARNING
            message = (
                f"Model may fit on {device} but leaves little headroom "
                f"({estimated_peak / (1024**3):.2f} GB of {available_gb:.2f} GB). "
                f"Other processes or larger batch sizes could cause OOM."
            )
        else:
            level = HardwareWarningLevel.OK
            message = f"Model fits comfortably on {device}."

        return HardwareReport(
            level=level,
            message=message,
            available_ram_gb=resources["available_ram_gb"],
            available_vram_gb=vram_gb,
            estimated_model_bytes=model_size_bytes,
            estimated_peak_bytes=estimated_peak,
            device_used=device,
        )

    # ---- internals ----

    def _get_gpu_info(self) -> list[dict[str, Any]]:
        gpus: list[dict[str, Any]] = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    gpus.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_vram_gb": total_bytes / (1024 ** 3),
                        "free_vram_gb": free_bytes / (1024 ** 3),
                    })
        except ImportError:
            pass
        return gpus

    def _select_device(self, resources: dict[str, Any], prefer_gpu: bool) -> tuple[str, int]:
        if prefer_gpu and resources["gpus"]:
            best_gpu = max(resources["gpus"], key=lambda g: g["free_vram_gb"])
            return f"cuda:{best_gpu['index']}", int(best_gpu["free_vram_gb"] * (1024 ** 3))

        available_ram_bytes = int(resources["available_ram_gb"] * (1024 ** 3))
        return "cpu", available_ram_bytes