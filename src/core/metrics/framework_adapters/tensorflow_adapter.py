from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ...exceptions.adapters import TensorflowUnbuiltModelError

@dataclass
class TensorflowParamBreakdown:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    is_built: bool
    mixed_precision_policy: str | None
    dtype_breakdown: dict[str, int] = field(default_factory=dict)


class TensorflowAdapter:
    def matches(self, model: Any) -> bool:
        try:
            import tensorflow as tf
        except ImportError:
            return False
        return isinstance(model, (tf.keras.Model, tf.Module)) or self._is_saved_model(model)

    def framework_name(self) -> str:
        return "tensorflow"

    def count_parameters(self, model: Any) -> tuple[int, int]:
        resolved = self._resolve(model)
        self._ensure_built(resolved)

        total = resolved.count_params()
        trainable = sum(int(tf_shape_num_elements(w)) for w in resolved.trainable_weights)
        return total, trainable

    def model_size_bytes(self, model: Any) -> int:
        resolved = self._resolve(model)
        self._ensure_built(resolved)

        total_bytes = 0
        for w in resolved.weights:
            total_bytes += w.numpy().nbytes
        return total_bytes

    def full_breakdown(self, model: Any) -> TensorflowParamBreakdown:
        resolved = self._resolve(model)
        is_built = self._is_built(resolved)

        if not is_built:
            return TensorflowParamBreakdown(
                total_params=0, trainable_params=0, non_trainable_params=0,
                is_built=False, mixed_precision_policy=None,
            )

        total, trainable = self.count_parameters(resolved)
        dtype_counts: dict[str, int] = {}
        for w in resolved.weights:
            key = str(w.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + int(tf_shape_num_elements(w))

        policy = None
        if hasattr(resolved, "dtype_policy"):
            policy = resolved.dtype_policy.name

        return TensorflowParamBreakdown(
            total_params=total,
            trainable_params=trainable,
            non_trainable_params=total - trainable,
            is_built=True,
            mixed_precision_policy=policy,
            dtype_breakdown=dtype_counts,
        )

    # ---- internals ----

    def _is_saved_model(self, model: Any) -> bool:
        return type(model).__name__ == "AutoTrackable" and hasattr(model, "signatures")

    def _resolve(self, model: Any) -> Any:
        if hasattr(model, "_original_model"):
            return model._original_model
        return model

    def _is_built(self, model: Any) -> bool:
        if self._is_saved_model(model):
            return True
        return getattr(model, "built", True)

    def _ensure_built(self, model: Any) -> None:
        if self._is_saved_model(model):
            return
        if not getattr(model, "built", True):
            raise TensorflowUnbuiltModelError(
                "Model has not been built yet — call model.build(input_shape) "
                "or run a forward pass before counting parameters."
            )


def tf_shape_num_elements(tensor: Any) -> int:
    shape = tensor.shape
    if hasattr(shape, "num_elements"):
        n = shape.num_elements()
        return n if n is not None else 0
    result = 1
    for dim in shape:
        result *= dim if dim is not None else 0
    return result