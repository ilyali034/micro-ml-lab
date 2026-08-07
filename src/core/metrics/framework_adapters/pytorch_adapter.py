from typing import Any

class PytorchAdapter:
    def matches(self, model: Any) -> bool:
        try:
            import torch.nn as nn
            return isinstance(model, nn.Module)
        except ImportError:
            return False

    def count_parameters(self, model: Any) -> tuple[int, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def model_size_bytes(self, model: Any) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters())

    def framework_name(self) -> str:
        return "pytorch"