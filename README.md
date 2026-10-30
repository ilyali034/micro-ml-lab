# Micro ML Lab

A local-first machine learning workbench for model inspection, inference and performance experiments.

## Micro ML Lab Studio

The new Streamlit interface provides:
- local LLM and BERT inference;
- masked language modeling, text generation, classification and embeddings;
- latency, throughput and GPU memory benchmarks;
- experiment history and JSON export;
- CPU, RAM, CUDA and VRAM overview.

### Run the Studio

Install a CUDA-compatible PyTorch build for your system first, then:

    python -m pip install -r studio-requirements.txt
    streamlit run studio/app.py

See [STUDIO.md](STUDIO.md) for Windows setup, model loading examples and benchmark methodology.

The Studio is an optional interface and does not replace the existing FastAPI/domain modules.
