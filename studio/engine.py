"""Local inference and benchmarking engine for Micro ML Lab Studio.

Models are loaded lazily. Nothing is uploaded to a remote inference service.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
HISTORY_FILE = RESULTS_DIR / "studio_benchmarks.jsonl"


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "ram_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append({
                "name": props.name,
                "vram_total_gb": round(props.total_memory / 1024**3, 2),
                "vram_allocated_gb": round(torch.cuda.memory_allocated(index) / 1024**3, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved(index) / 1024**3, 2),
            })
    return info


def discover_local_models(roots: list[str] | None = None) -> list[dict[str, str]]:
    """Find Hugging Face model folders without recursively scanning whole disks."""
    candidates = roots or [
        str(Path.home() / ".cache" / "huggingface" / "hub"),
        "./models",
        "./ml_models",
    ]
    found: dict[str, dict[str, str]] = {}
    for root in candidates:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        try:
            for config in base.glob("**/config.json"):
                # Avoid traversing deeply nested snapshot duplicates.
                if len(config.parts) - len(base.parts) > 7:
                    continue
                folder = config.parent
                key = str(folder.resolve())
                found[key] = {"name": folder.name, "path": key}
        except (OSError, PermissionError):
            continue
    return sorted(found.values(), key=lambda item: item["name"].lower())


def _dtype(name: str):
    return {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def load_model(model_id: str, task: str, dtype_name: str = "auto", device: str = "auto"):
    """Load a model and tokenizer. task: causal, masked, classification, embeddings."""
    if not model_id.strip():
        raise ValueError("Укажи локальный путь к модели или Hugging Face ID.")
    target = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    kwargs: dict[str, Any] = {"torch_dtype": _dtype(dtype_name), "low_cpu_mem_usage": True}
    if target == "cuda":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if task == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    elif task == "masked":
        model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    elif task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    elif task == "embeddings":
        model = AutoModel.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError(f"Неизвестная задача: {task}")
    if target != "cuda":
        model = model.to(target)
    model.eval()
    return model, tokenizer, target


def run_inference(model, tokenizer, task: str, text: str, max_new_tokens: int = 128) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    try:
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
    except StopIteration:
        pass

    with torch.inference_mode():
        if task == "causal":
            prompt_tokens = inputs["input_ids"].shape[-1]
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output[0, prompt_tokens:]
            result = tokenizer.decode(generated, skip_special_tokens=True)
            output_tokens = int(generated.numel())
        elif task == "masked":
            output = model(**inputs).logits
            token_ids = inputs["input_ids"][0].clone()
            mask_id = tokenizer.mask_token_id
            if mask_id is None:
                raise ValueError("У токенизатора нет mask_token_id. Выбери masked language model.")
            positions = (token_ids == mask_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                raise ValueError(f"Добавь токен {tokenizer.mask_token} в текст.")
            pos = int(positions[0])
            best = output[0, pos].topk(5)
            result = "\n".join(
                f"{tokenizer.decode([idx]).strip()}  —  {float(score):.3f}"
                for score, idx in zip(best.values, best.indices)
            )
            output_tokens = 1
        elif task == "classification":
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            top = probs.topk(min(5, probs.numel()))
            labels = model.config.id2label or {}
            result = "\n".join(
                f"{labels.get(int(idx), f'LABEL_{int(idx)}')}  —  {float(score):.4f}"
                for score, idx in zip(top.values, top.indices)
            )
            output_tokens = 1
        else:
            hidden = model(**inputs).last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                embedding = hidden.mean(dim=1)
            else:
                expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
                embedding = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
            vector = embedding[0].detach().float().cpu()
            result = f"Вектор размерности: {vector.numel()}\nПервые 16 значений: {vector[:16].tolist()}"
            output_tokens = 1

    elapsed = time.perf_counter() - started
    return {
        "output": result,
        "latency_seconds": elapsed,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
    }


def benchmark(model, tokenizer, task: str, prompt: str, iterations: int = 5,
              max_new_tokens: int = 64) -> dict[str, Any]:
    """Warm up once, then measure repeated inference and memory."""
    run_inference(model, tokenizer, task, prompt, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    total_tokens = 0
    for _ in range(iterations):
        result = run_inference(model, tokenizer, task, prompt, max_new_tokens)
        samples.append(result["latency_seconds"])
        total_tokens += result["output_tokens"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_vram = 0.0
    ordered = sorted(samples)
    mean = sum(samples) / len(samples)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "iterations": iterations,
        "mean_latency_ms": mean * 1000,
        "median_latency_ms": ordered[len(ordered) // 2] * 1000,
        "min_latency_ms": min(samples) * 1000,
        "mean_tokens_per_second": (total_tokens / sum(samples)) if sum(samples) else 0,
        "peak_vram_gb": peak_vram,
        "samples_seconds": samples,
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def benchmark_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    rows = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-200:]
