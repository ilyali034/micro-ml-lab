from __future__ import annotations

import gc
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ..hardware.capability_checker import HardwareCapabilityChecker
from ..metrics.framework_adapters.registry import AdapterRegistry

ModelTask = Literal["causal", "masked_lm", "classification", "embeddings"]
ModelDevice = Literal["auto", "cpu", "cuda"]
ModelDtype = Literal["auto", "float16", "bfloat16", "float32"]

logger = logging.getLogger(__name__)


class RuntimeErrorBase(Exception):
    """Base exception for local inference runtime."""


class ModelNotLoadedError(RuntimeErrorBase):
    pass


class ModelLoadError(RuntimeErrorBase):
    pass


class InferenceError(RuntimeErrorBase):
    pass


@dataclass(slots=True)
class LoadedModel:
    model_id: str
    task: ModelTask
    device: str
    dtype: str
    model: Any
    tokenizer: Any
    loaded_at: str


class LocalInferenceEngine:
    """
    Single-active-model runtime for local Transformers inference.

    The single-model policy is intentional: it avoids silently keeping several
    large checkpoints in a small GPU's VRAM. Model loading and inference are
    serialized by a re-entrant lock. API handlers should call this class via
    asyncio.to_thread so model work never blocks the ASGI event loop.
    """

    def __init__(
        self,
        benchmark_dir: str | Path = "artifacts/benchmarks",
        allow_remote_models: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._active: LoadedModel | None = None
        self._benchmark_dir = Path(benchmark_dir)
        self._benchmark_dir.mkdir(parents=True, exist_ok=True)
        self._allow_remote_models = allow_remote_models
        self._trust_remote_code = trust_remote_code
        self._adapter_registry = AdapterRegistry()
        self._hardware = HardwareCapabilityChecker()

    @property
    def is_loaded(self) -> bool:
        return self._active is not None

    def status(self) -> dict[str, Any]:
        resources = self._hardware.get_system_resources()
        active = self._active
        return {
            "state": "loaded" if active else "idle",
            "active_model": self.model_info() if active else None,
            "resources": resources,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }

    def model_info(self) -> dict[str, Any]:
        with self._lock:
            if self._active is None:
                raise ModelNotLoadedError("No model is loaded.")
            active = self._active
            adapter = self._adapter_registry.resolve(active.model)
            total, trainable = adapter.count_parameters(active.model)
            return {
                "model_id": active.model_id,
                "task": active.task,
                "device": active.device,
                "dtype": active.dtype,
                "loaded_at": active.loaded_at,
                "framework": adapter.framework_name(),
                "total_parameters": total,
                "trainable_parameters": trainable,
                "model_size_bytes": adapter.model_size_bytes(active.model),
                "model_class": type(active.model).__name__,
                "config": {
                    "model_type": getattr(active.model.config, "model_type", None),
                    "architectures": getattr(active.model.config, "architectures", None),
                    "hidden_size": getattr(active.model.config, "hidden_size", None),
                    "num_hidden_layers": getattr(active.model.config, "num_hidden_layers", None),
                    "vocab_size": getattr(active.model.config, "vocab_size", None),
                    "max_position_embeddings": getattr(active.model.config, "max_position_embeddings", None),
                },
            }

    def load(
        self,
        model_id: str,
        task: ModelTask = "causal",
        device: ModelDevice = "auto",
        dtype: ModelDtype = "auto",
        local_files_only: bool = True,
    ) -> dict[str, Any]:
        model_id = model_id.strip()
        if not model_id:
            raise ModelLoadError("model_id must be a non-empty local path or Hugging Face ID.")
        if task not in {"causal", "masked_lm", "classification", "embeddings"}:
            raise ModelLoadError(f"Unsupported task: {task}")
        if local_files_only is False and not self._allow_remote_models:
            raise ModelLoadError("Remote model downloads are disabled by runtime configuration.")

        with self._lock:
            # Keep only one checkpoint in memory: free VRAM before replacement.
            previous = self._active
            self._active = None
            if previous is not None:
                del previous
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            new_model = None
            new_tokenizer = None
            try:
                target_device = self._resolve_device(device)
                torch_dtype = self._resolve_dtype(dtype)
                logger.info("Loading %s for task=%s on %s", model_id, task, target_device)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_fast=True,
                    local_files_only=local_files_only,
                    trust_remote_code=self._trust_remote_code,
                )
                config = AutoConfig.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                    trust_remote_code=self._trust_remote_code,
                )
                model_class = {
                    "causal": AutoModelForCausalLM,
                    "masked_lm": AutoModelForMaskedLM,
                    "classification": AutoModelForSequenceClassification,
                    "embeddings": AutoModel,
                }[task]

                kwargs: dict[str, Any] = {
                    "config": config,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": self._trust_remote_code,
                    "local_files_only": local_files_only,
                }
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype

                if target_device == "cuda":
                    kwargs["device_map"] = "auto"
                    kwargs["max_memory"] = self._max_memory_map()
                model = model_class.from_pretrained(model_id, **kwargs)
                if target_device == "cpu":
                    model = model.to("cpu")
                model.eval()

                if task == "causal" and tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token

                new_model, new_tokenizer = model, tokenizer
                old = self._active
                self._active = LoadedModel(
                    model_id=model_id,
                    task=task,
                    device=target_device,
                    dtype=str(next(model.parameters()).dtype),
                    model=model,
                    tokenizer=tokenizer,
                    loaded_at=datetime.now(timezone.utc).isoformat(),
                )
                if old is not None:
                    del old
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Loaded model %s successfully", model_id)
                return self.model_info()
            except Exception as exc:
                if new_model is not None:
                    del new_model
                if new_tokenizer is not None:
                    del new_tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.exception("Failed to load model %s", model_id)
                if isinstance(exc, RuntimeErrorBase):
                    raise
                raise ModelLoadError(f"Could not load '{model_id}': {exc}") from exc

    def unload(self) -> dict[str, str]:
        with self._lock:
            if self._active is None:
                return {"state": "idle", "message": "No model was loaded."}
            model_id = self._active.model_id
            self._active = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("Unloaded model %s", model_id)
            return {"state": "idle", "message": f"Unloaded {model_id}."}

    def infer(
        self,
        text: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        max_input_tokens: int = 4096,
    ) -> dict[str, Any]:
        if not text.strip():
            raise InferenceError("Input text must not be empty.")
        with self._lock:
            active = self._require_model()
            started = time.perf_counter()
            try:
                encoded = active.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                )
                input_tokens = int(encoded["input_ids"].shape[-1])
                input_device = self._input_device(active.model)
                encoded = {key: value.to(input_device) for key, value in encoded.items()}

                with torch.inference_mode():
                    if active.task == "causal":
                        generation_kwargs: dict[str, Any] = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": do_sample,
                            "repetition_penalty": repetition_penalty,
                            "pad_token_id": active.tokenizer.pad_token_id,
                            "eos_token_id": active.tokenizer.eos_token_id,
                        }
                        if do_sample:
                            generation_kwargs.update({
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                            })
                        output = active.model.generate(**encoded, **generation_kwargs)
                        generated = output[0, input_tokens:]
                        result: Any = active.tokenizer.decode(
                            generated.detach().cpu().tolist(),
                            skip_special_tokens=True,
                        )
                        output_tokens = int(generated.numel())
                    elif active.task == "masked_lm":
                        mask_id = active.tokenizer.mask_token_id
                        if mask_id is None:
                            raise InferenceError("This tokenizer has no mask token.")
                        mask_positions = (encoded["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
                        if mask_positions.numel() == 0:
                            raise InferenceError(
                                f"Input must contain {active.tokenizer.mask_token!r}."
                            )
                        logits = active.model(**encoded).logits[0]
                        position = int(mask_positions[0])
                        scores = torch.softmax(logits[position], dim=-1)
                        values, indices = scores.topk(5)
                        result = [
                            {
                                "token": active.tokenizer.decode([int(index)]).strip(),
                                "score": float(score),
                            }
                            for score, index in zip(values.cpu(), indices.cpu())
                        ]
                        output_tokens = input_tokens
                    elif active.task == "classification":
                        logits = active.model(**encoded).logits[0]
                        scores = torch.softmax(logits.float(), dim=-1)
                        values, indices = scores.topk(min(5, scores.numel()))
                        labels = getattr(active.model.config, "id2label", {}) or {}
                        result = [
                            {
                                "label": labels.get(int(index), f"LABEL_{int(index)}"),
                                "score": float(score),
                            }
                            for score, index in zip(values.cpu(), indices.cpu())
                        ]
                        output_tokens = input_tokens
                    else:
                        hidden = active.model(**encoded).last_hidden_state
                        mask = encoded.get("attention_mask")
                        if mask is None:
                            embedding = hidden.mean(dim=1)
                        else:
                            weights = mask.unsqueeze(-1).to(hidden.dtype)
                            embedding = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                        vector = embedding[0].detach().float().cpu()
                        result = {
                            "dimension": int(vector.numel()),
                            "embedding": vector.tolist(),
                        }
                        output_tokens = input_tokens

                self._synchronize(active.model)
                elapsed = time.perf_counter() - started
                return {
                    "model_id": active.model_id,
                    "task": active.task,
                    "output": result,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_seconds": elapsed,
                    "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
                }
            except RuntimeErrorBase:
                raise
            except Exception as exc:
                logger.exception("Inference failed for %s", active.model_id)
                raise InferenceError(str(exc)) from exc

    def benchmark(
        self,
        prompt: str,
        iterations: int = 5,
        warmup_runs: int = 1,
        max_new_tokens: int = 64,
        **generation_options: Any,
    ) -> dict[str, Any]:
        if iterations < 1 or iterations > 100:
            raise ValueError("iterations must be between 1 and 100.")
        with self._lock:
            active = self._require_model()
            for _ in range(warmup_runs):
                self.infer(prompt, max_new_tokens=max_new_tokens, **generation_options)

            cuda_devices = self._cuda_devices(active.model)
            for index in cuda_devices:
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)

            samples: list[dict[str, Any]] = []
            for _ in range(iterations):
                samples.append(
                    self.infer(prompt, max_new_tokens=max_new_tokens, **generation_options)
                )

            for index in cuda_devices:
                torch.cuda.synchronize(index)

            latencies = [sample["latency_seconds"] for sample in samples]
            ordered = sorted(latencies)
            total_time = sum(latencies)
            total_tokens = sum(sample["output_tokens"] for sample in samples)
            peak_allocated = max(
                (torch.cuda.max_memory_allocated(index) for index in cuda_devices),
                default=0,
            )
            peak_reserved = max(
                (torch.cuda.max_memory_reserved(index) for index in cuda_devices),
                default=0,
            )
            record = {
                "id": f"bench-{int(time.time() * 1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_id": active.model_id,
                "task": active.task,
                "device": active.device,
                "dtype": active.dtype,
                "iterations": iterations,
                "warmup_runs": warmup_runs,
                "mean_latency_ms": (total_time / iterations) * 1000,
                "median_latency_ms": ordered[len(ordered) // 2] * 1000,
                "min_latency_ms": min(latencies) * 1000,
                "max_latency_ms": max(latencies) * 1000,
                "throughput_tokens_per_second": total_tokens / total_time if total_time else 0.0,
                "total_output_tokens": total_tokens,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "samples_seconds": latencies,
            }
            self._write_benchmark(record)
            return record

    def benchmark_history(self, limit: int = 100) -> list[dict[str, Any]]:
        history_file = self._benchmark_dir / "runs.jsonl"
        if not history_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        with history_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed benchmark history row.")
        return rows[-limit:]

    def _write_benchmark(self, record: dict[str, Any]) -> None:
        history_file = self._benchmark_dir / "runs.jsonl"
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _require_model(self) -> LoadedModel:
        if self._active is None:
            raise ModelNotLoadedError("Load a model before running inference.")
        return self._active

    @staticmethod
    def _resolve_device(device: ModelDevice) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but PyTorch cannot access a CUDA device.")
        return device

    @staticmethod
    def _resolve_dtype(dtype: ModelDtype):
        return {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    @staticmethod
    def _max_memory_map() -> dict[Any, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        ram_available = psutil.virtual_memory().available
        return {
            0: max(int(free_bytes * 0.88), 1),
            "cpu": max(int(ram_available * 0.75), 1),
        }

    @staticmethod
    def _input_device(model: Any) -> torch.device:
        try:
            return model.get_input_embeddings().weight.device
        except Exception:
            return next(model.parameters()).device

    @staticmethod
    def _cuda_devices(model: Any) -> list[int]:
        devices: set[int] = set()
        for parameter in model.parameters():
            if parameter.device.type == "cuda":
                devices.add(parameter.device.index or 0)
        return sorted(devices)

    @staticmethod
    def _synchronize(model: Any) -> None:
        for index in LocalInferenceEngine._cuda_devices(model):
            torch.cuda.synchronize(index)
