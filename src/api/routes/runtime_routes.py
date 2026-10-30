from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from core.inference.runtime import (
    InferenceError,
    LocalInferenceEngine,
    ModelLoadError,
    ModelNotLoadedError,
)
from shared.dtos.runtime import BenchmarkRequest, InferenceRequest, LoadModelRequest


def get_inference_engine(request: Request) -> LocalInferenceEngine:
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Local inference runtime is not initialized.")
    return engine

router = APIRouter(prefix="/runtime", tags=["local-runtime"])


def _raise_runtime_error(exc: Exception) -> None:
    if isinstance(exc, ModelNotLoadedError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (ModelLoadError, InferenceError, ValueError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail="Local runtime error.") from exc


@router.get("/status")
async def runtime_status(
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    return await asyncio.to_thread(engine.status)


@router.get("/model")
async def active_model(
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    try:
        return await asyncio.to_thread(engine.model_info)
    except Exception as exc:
        _raise_runtime_error(exc)


@router.post("/load")
async def load_model(
    payload: LoadModelRequest,
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    try:
        return await asyncio.to_thread(
            engine.load,
            payload.model_id,
            payload.task,
            payload.device,
            payload.dtype,
            payload.local_files_only,
        )
    except Exception as exc:
        _raise_runtime_error(exc)


@router.post("/unload")
async def unload_model(
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    return await asyncio.to_thread(engine.unload)


@router.post("/infer")
async def run_inference(
    payload: InferenceRequest,
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    try:
        return await asyncio.to_thread(
            engine.infer,
            payload.text,
            payload.max_new_tokens,
            payload.do_sample,
            payload.temperature,
            payload.top_p,
            payload.top_k,
            payload.repetition_penalty,
            payload.max_input_tokens,
        )
    except Exception as exc:
        _raise_runtime_error(exc)


@router.post("/benchmark")
async def run_benchmark(
    payload: BenchmarkRequest,
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
):
    try:
        return await asyncio.to_thread(
            engine.benchmark,
            payload.prompt,
            payload.iterations,
            payload.warmup_runs,
            payload.max_new_tokens,
            do_sample=payload.do_sample,
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            repetition_penalty=payload.repetition_penalty,
        )
    except Exception as exc:
        _raise_runtime_error(exc)


@router.get("/benchmarks")
async def benchmark_history(
    engine: Annotated[LocalInferenceEngine, Depends(get_inference_engine)],
    limit: int = Query(default=50, ge=1, le=500),
):
    return await asyncio.to_thread(engine.benchmark_history, limit)
