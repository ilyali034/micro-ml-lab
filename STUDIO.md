# Micro ML Lab Studio

Micro ML Lab has two local processes with a clean separation:

- **FastAPI inference runtime** owns model loading, device placement, inference, memory cleanup and benchmark execution.
- **Streamlit Studio** is only a UI client. It sends requests to the runtime API; it does not load models itself.

## 1. Install backend dependencies

Use the Python environment that has the CUDA-enabled PyTorch build you intend to use. Install PyTorch for your OS/CUDA combination first if it is not already installed.

PowerShell from the repository root:

    python -m pip install -r requirements.txt

The runtime needs Transformers, Accelerate and psutil in addition to the existing API dependencies.

## 2. Start the local backend

PowerShell, from the repository root:

    $env:PYTHONPATH = "$PWD\src"
    python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

API documentation is available at http://127.0.0.1:8000/docs.

The runtime listens on localhost by default. Do not expose this unauthenticated development API to an untrusted network.

## 3. Start the Studio UI

Open a second PowerShell window in the repository root:

    python -m pip install -r studio-requirements.txt
    streamlit run studio/app.py

The default backend URL is http://127.0.0.1:8000. It can be changed in the sidebar or with MICRO_ML_API_URL.

## Runtime API

- GET /runtime/status — hardware and active model status.
- POST /runtime/load — load one checkpoint.
- GET /runtime/model — active model metadata and parameter statistics.
- POST /runtime/unload — release model references and CUDA cache.
- POST /runtime/infer — run the selected task.
- POST /runtime/benchmark — warm-up plus repeated measurements.
- GET /runtime/benchmarks — local benchmark history.

Example model load:

    curl -X POST http://127.0.0.1:8000/runtime/load -H "Content-Type: application/json" -d "{\"model_id\":\"D:/ml_models/Qwen2.5-0.5B-Instruct\",\"task\":\"causal\",\"device\":\"auto\",\"dtype\":\"float16\",\"local_files_only\":true}"

## Supported Transformers tasks

- causal — AutoModelForCausalLM text generation.
- masked_lm — AutoModelForMaskedLM fill-mask inference.
- classification — AutoModelForSequenceClassification top-label prediction.
- embeddings — AutoModel plus attention-mask-aware mean pooling.

A classification checkpoint must have a trained classification head. A plain BERT base checkpoint is not automatically a sentiment classifier.

## Runtime design and safety

- One active model at a time. Loading another checkpoint first unloads the current one to avoid keeping multiple models in limited VRAM.
- Inference and benchmark calls are serialized by a runtime lock.
- Heavy model operations run in worker threads, not on the FastAPI event loop.
- Remote model downloads are disabled per request by default (`local_files_only=true`). Enable downloads explicitly in Model Studio.
- `trust_remote_code` is disabled in the runtime.
- CUDA placement uses Accelerate `device_map=auto` and a memory budget based on currently free VRAM and RAM.
- Benchmark history is appended to `artifacts/benchmarks/runs.jsonl`.
- Peak CUDA allocated/reserved memory is PyTorch allocator telemetry, not total process memory reported by the NVIDIA driver.

## Benchmark interpretation

For causal models, throughput is generated tokens divided by total measured generation time. For encoder tasks, throughput uses processed input tokens. The warm-up phase is excluded from samples. Compare runs only when model revision, prompt, task, dtype, device, generation settings and iteration count are held constant.

## Tests

Install test tooling and run the runtime lifecycle tests from the repository root:

    python -m pip install -r requirements-dev.txt
    $env:PYTHONPATH = "$PWD\src"
    python -m pytest tests/test_local_inference_runtime.py -q

These tests cover idle lifecycle behavior and input validation. A full model/GPU smoke test still needs to be run on the target machine.

## Current scope

This runtime is the inference foundation. Fine-tuning/LoRA, datasets, RAG/vector stores, GGUF/llama.cpp, streaming generation and multi-model scheduling are separate follow-up modules.
