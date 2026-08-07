from typing import Any
from .protocols import FrameworkAdapter
from .pytorch_adapter import PytorchAdapter
from .sklearn_adapter import SklearnAdapter
from .tensorflow_adapter import TensorflowAdapter

class AdapterRegistry:
    def __init__(self, adapters: list[FrameworkAdapter] | None = None):
        self._adapters = adapters or [PytorchAdapter(), SklearnAdapter(), TensorflowAdapter()]

    def resolve(self, model: Any) -> FrameworkAdapter:
        for adapter in self._adapters:
            if adapter.matches(model):
                return adapter
        raise TypeError(f"No adapter found for model type: {type(model)}")