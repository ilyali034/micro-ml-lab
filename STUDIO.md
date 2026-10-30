# Micro ML Lab Studio

A local-first workbench for Hugging Face LLM and BERT experiments.

## Modules
- Overview: hardware summary, active model state and recent benchmark runs.
- Model Studio: load local folders or Hugging Face IDs; run causal generation, masked language modeling, sequence classification and mean-pooled embeddings.
- Benchmark Lab: warm-up plus repeated measurements of mean/median/min latency, throughput and peak CUDA allocated memory.
- Experiments: inspect and export benchmark history.
- System Monitor: CPU/RAM/GPU/VRAM and runtime information.

Models execute in the local Python process. No third-party inference API is used. A remote Hugging Face model ID may trigger a model download if the files are not cached.

## Quick start (Windows PowerShell)

Use the same Python environment where your CUDA-enabled PyTorch is installed.

    git pull
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    # Install a CUDA-compatible PyTorch build first: https://pytorch.org/get-started/locally/
    python -m pip install -r studio-requirements.txt
    streamlit run studio/app.py

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Loading models
In Model Studio, enter a local folder such as D:/ml_models/Qwen2.5-0.5B-Instruct or a Hugging Face ID such as Qwen/Qwen2.5-0.5B-Instruct.

Choose the matching task:
- causal generation → AutoModelForCausalLM;
- masked language modeling → AutoModelForMaskedLM;
- classification → AutoModelForSequenceClassification;
- embeddings → AutoModel with attention-mask-aware mean pooling.

For classification, use a checkpoint fine-tuned for sequence classification. A base BERT checkpoint is not automatically a sentiment classifier.

## Benchmark methodology
The first pass is a warm-up and excluded from samples. Latency is measured per complete inference call. For causal models, throughput is generated tokens divided by generation time. Peak VRAM is PyTorch CUDA allocated memory, not total driver usage. Records append to results/studio_benchmarks.jsonl.

For meaningful comparisons, keep model revision, prompt, dtype, device, generation length and iteration count unchanged.

## Current scope
This is the first usable Studio layer alongside the existing FastAPI/domain code. It does not modify the current database schema or replace the API. Training/fine-tuning, dataset management, RAG/vector stores, GGUF/llama.cpp and live GPU telemetry are planned follow-up modules.
