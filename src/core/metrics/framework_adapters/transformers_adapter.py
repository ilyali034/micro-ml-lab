from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformersParamBreakdown:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    embedding_params: int
    is_sharded: bool          # accelerate device_map="auto" / offload
    is_peft: bool
    quantization_method: str | None   # "gptq" | "awq" | "bitsandbytes" | None
    dtype_breakdown: dict[str, int] = field(default_factory=dict)


class TransformersAdapter:
    """
    Production adapter for HuggingFace `transformers` models.
    Not merged into PytorchAdapter because HF models need library-specific
    handling: built-in num_parameters(), PEFT wrapping, accelerate sharding,
    GPTQ/AWQ quantization metadata.
    """

    def matches(self, model: Any) -> bool:
        try:
            from transformers import PreTrainedModel
        except ImportError:
            return False
        return isinstance(model, PreTrainedModel) or self._is_peft_wrapped(model)

    def framework_name(self) -> str:
        return "transformers"

    def count_parameters(self, model: Any) -> tuple[int, int]:
        base_model, is_peft = self._resolve_base(model)

        if is_peft:
            trainable, total = model.get_nb_trainable_parameters()
        else:
            total = base_model.num_parameters(only_trainable=False)
            trainable = base_model.num_parameters(only_trainable=True)

        return total, trainable

    def model_size_bytes(self, model: Any) -> int:
        base_model, _ = self._resolve_base(model)
        quant = self._detect_quantization(base_model)

        if quant is not None:
            return self._estimate_quantized_bytes(base_model, quant)

        return sum(
            p.numel() * p.element_size()
            for p in base_model.parameters()
            if p.device.type != "meta"
        )

    def full_breakdown(self, model: Any) -> TransformersParamBreakdown:
        base_model, is_peft = self._resolve_base(model)
        total, trainable = self.count_parameters(model)

        embedding_params = self._count_embedding_params(base_model)
        quant = self._detect_quantization(base_model)
        is_sharded = self._is_sharded(base_model)

        dtype_counts: dict[str, int] = {}
        for p in base_model.parameters():
            key = str(p.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + p.numel()

        return TransformersParamBreakdown(
            total_params=total,
            trainable_params=trainable,
            non_trainable_params=total - trainable,
            embedding_params=embedding_params,
            is_sharded=is_sharded,
            is_peft=is_peft,
            quantization_method=quant,
            dtype_breakdown=dtype_counts,
        )

    # ---- internals ----

    def _is_peft_wrapped(self, model: Any) -> bool:
        try:
            from peft import PeftModel
            return isinstance(model, PeftModel)
        except ImportError:
            return False

    def _resolve_base(self, model: Any) -> tuple[Any, bool]:
        """Возвращает (base_transformers_model, is_peft)."""
        if self._is_peft_wrapped(model):
            return model, True
        return model, False

    def _count_embedding_params(self, model: Any) -> int:
        try:
            emb = model.get_input_embeddings()
            return emb.weight.numel() if emb is not None else 0
        except Exception:
            return 0

    def _detect_quantization(self, model: Any) -> str | None:
        quant_config = getattr(model, "quantization_config", None) or getattr(
            getattr(model, "config", None), "quantization_config", None
        )
        if quant_config is not None:
            method = getattr(quant_config, "quant_method", None)
            if method:
                return str(method)

        for m in model.modules():
            cls_name = type(m).__name__
            if cls_name in ("Linear8bitLt", "Linear4bit"):
                return "bitsandbytes"
            if "GPTQ" in cls_name:
                return "gptq"
            if "AWQ" in cls_name.upper():
                return "awq"
        return None

    def _is_sharded(self, model: Any) -> bool:
        return hasattr(model, "hf_device_map") and len(set(model.hf_device_map.values())) > 1

    def _estimate_quantized_bytes(self, model: Any, quant_method: str) -> int:
        bits_per_param = {"bitsandbytes": 8, "gptq": 4, "awq": 4}.get(quant_method, 8)
        total_params = model.num_parameters(only_trainable=False)
        return (total_params * bits_per_param) // 8