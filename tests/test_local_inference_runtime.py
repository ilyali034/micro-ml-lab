import pytest

from core.inference.runtime import (
    LocalInferenceEngine,
    ModelNotLoadedError,
)


def test_unload_is_safe_when_no_model_is_loaded(tmp_path):
    engine = LocalInferenceEngine(benchmark_dir=tmp_path)
    result = engine.unload()

    assert result["state"] == "idle"
    assert engine.is_loaded is False


def test_inference_requires_a_loaded_model(tmp_path):
    engine = LocalInferenceEngine(benchmark_dir=tmp_path)

    with pytest.raises(ModelNotLoadedError):
        engine.infer("hello")


def test_benchmark_validates_iteration_count_before_inference(tmp_path):
    engine = LocalInferenceEngine(benchmark_dir=tmp_path)

    with pytest.raises(ValueError, match="iterations"):
        engine.benchmark("hello", iterations=0)


def test_empty_benchmark_history(tmp_path):
    engine = LocalInferenceEngine(benchmark_dir=tmp_path)

    assert engine.benchmark_history() == []
