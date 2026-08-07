from typing import Any
from .protocols import FrameworkAdapter
from .pytorch_adapter import PytorchAdapter
from .transformers_adapter import TransformersAdapter
from .sklearn_adapter import SklearnAdapter
from .tensorflow_adapter import TensorflowAdapter


class NoMatchingAdapterError(Exception):
    pass


class AdapterRegistry:
    def __init__(self, adapters: list[FrameworkAdapter] | None = None):
        self._adapters = adapters or [
            TransformersAdapter(),
            PytorchAdapter(),
            SklearnAdapter(),
            TensorflowAdapter(),
        ]

    def resolve(self, model: Any) -> FrameworkAdapter:
        for adapter in self._adapters:
            if adapter.matches(model):
                return adapter
        raise NoMatchingAdapterError(f"No adapter found for model type: {type(model)}")