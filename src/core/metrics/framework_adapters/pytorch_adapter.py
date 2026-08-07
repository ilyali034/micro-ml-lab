from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PytorchParamBreakdown:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    shared_params: int         
    buffer_count: int
    buffer_bytes: int
    is_meta_device: bool
    is_quantized: bool
    dtype_breakdown: dict[str, int] = field(default_factory=dict)  # {"torch.float16": 1234, ...}


class PytorchAdapter:
    """
    Production adapter for PyTorch models. Handles DDP/FSDP wrapping,
    torch.compile, PEFT/LoRA, meta-device models, quantization, and
    tied/shared weights.
    """

    def matches(self, model: Any) -> bool:
        try:
            import torch.nn as nn
        except ImportError:
            return False
        return isinstance(model, nn.Module) or self._is_compiled_module(model)

    def framework_name(self) -> str:
        return "pytorch"

    # ---- public API expected by FrameworkAdapter protocol ----

    def count_parameters(self, model: Any) -> tuple[int, int]:
        real_model = self._unwrap(model)
        breakdown = self._breakdown(real_model)
        return breakdown.total_params, breakdown.trainable_params

    def model_size_bytes(self, model: Any) -> int:
        real_model = self._unwrap(model)
        breakdown = self._breakdown(real_model)
        if breakdown.is_meta_device:
            return self._theoretical_bytes(real_model)
        return self._actual_bytes(real_model) + breakdown.buffer_bytes

    def full_breakdown(self, model: Any) -> PytorchParamBreakdown:
        return self._breakdown(self._unwrap(model))


    def _unwrap(self, model: Any) -> Any:
        seen = set()
        current = model

        while id(current) not in seen:
            seen.add(id(current))

            if hasattr(current, "_orig_mod"):
                current = current._orig_mod
                continue

            cls_name = type(current).__name__
            if cls_name in ("DistributedDataParallel", "FullyShardedDataParallel") and hasattr(current, "module"):
                current = current.module
                continue

            break

        return current

    def _is_peft_model(self, model: Any) -> bool:
        try:
            from peft import PeftModel
            return isinstance(model, PeftModel)
        except ImportError:
            return False

    def _is_compiled_module(self, model: Any) -> bool:
        return hasattr(model, "_orig_mod")

    # ---- core counting ----

    def _breakdown(self, model: Any) -> PytorchParamBreakdown:
        import torch

        seen_param_ids: set[int] = set()
        total = 0
        trainable = 0
        shared = 0
        dtype_counts: dict[str, int] = {}
        is_meta = False

        for p in model.parameters():
            n = p.numel()
            dtype_key = str(p.dtype)
            dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + n

            if p.device.type == "meta":
                is_meta = True

            if id(p) in seen_param_ids:
                shared += n
                continue
            seen_param_ids.add(id(p))

            total += n
            if p.requires_grad:
                trainable += n

        non_trainable = total - trainable

        buffer_count = 0
        buffer_bytes = 0
        for b in model.buffers():
            buffer_count += b.numel()
            if b.device.type != "meta":
                buffer_bytes += b.numel() * b.element_size()

        is_quantized = self._detect_quantization(model)

        return PytorchParamBreakdown(
            total_params=total,
            trainable_params=trainable,
            non_trainable_params=non_trainable,
            shared_params=shared,
            buffer_count=buffer_count,
            buffer_bytes=buffer_bytes,
            is_meta_device=is_meta,
            is_quantized=is_quantized,
            dtype_breakdown=dtype_counts,
        )

    def _detect_quantization(self, model: Any) -> bool:
        try:
            import torch.nn as nn
            import torch.ao.nn.quantized as nnq
            for m in model.modules():
                if isinstance(m, nnq.Linear) or isinstance(m, nnq.Conv2d):
                    return True
        except ImportError:
            pass

        for m in model.modules():
            cls_name = type(m).__name__
            if cls_name in ("Linear8bitLt", "Linear4bit"):
                return True

        return False

    def _actual_bytes(self, model: Any) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type != "meta")

    def _theoretical_bytes(self, model: Any) -> int:
        import torch

        dtype_sizes = {
            torch.float32: 4, torch.float64: 8, torch.float16: 2,
            torch.bfloat16: 2, torch.int8: 1, torch.uint8: 1,
            torch.int32: 4, torch.int64: 8,
        }
        total = 0
        for p in model.parameters():
            size_bytes = dtype_sizes.get(p.dtype, 4) 
            total += p.numel() * size_bytes
        return total