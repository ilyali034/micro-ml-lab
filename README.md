# Micro ML Lab

Local-first ML workbench for model inspection, inference and performance experiments.

## Architecture

- `src/core/inference` — local Transformers inference runtime.
- `src/api/routes/runtime_routes.py` — runtime HTTP API.
- `src/core/hardware` — resource and memory capability inspection.
- `src/core/metrics/framework_adapters` — model/framework statistics.
- `studio` — Streamlit client UI. It talks to FastAPI and does not load models itself.

## Quick start

Install CUDA-enabled PyTorch for your system, then install project dependencies:

    python -m pip install -r requirements.txt

Start the backend from the repository root in PowerShell:

    $env:PYTHONPATH = "$PWD\src"
    python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

In a second terminal:

    python -m pip install -r studio-requirements.txt
    streamlit run studio/app.py

See [STUDIO.md](STUDIO.md) for API endpoints, model tasks, local model loading and benchmark methodology.
