from typing import Any

class TensorflowAdapter:
    def matches(self, model: Any) -> bool:
        try:
            import tensorflow as tf
            return isinstance(model, (tf.keras.Model, tf.Module))
        except ImportError:
            return False

    def count_parameters(self, model: Any) -> tuple[int, int]:
        total = model.count_params()
        trainable = sum(
            w.shape.num_elements() for w in model.trainable_weights
        )
        return total, trainable

    def model_size_bytes(self, model: Any) -> int:
        total_bytes = 0
        for w in model.weights:
            total_bytes += w.numpy().nbytes
        return total_bytes

    def framework_name(self) -> str:
        return "tensorflow"