from typing import Literal
from pydantic import BaseModel, Field


ModelTask = Literal["causal", "masked_lm", "classification", "embeddings"]
ModelDevice = Literal["auto", "cpu", "cuda"]
ModelDtype = Literal["auto", "float16", "bfloat16", "float32"]


class LoadModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=2048)
    task: ModelTask = "causal"
    device: ModelDevice = "auto"
    dtype: ModelDtype = "auto"
    local_files_only: bool = True


class InferenceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    do_sample: bool = False
    temperature: float = Field(default=0.7, gt=0, le=5)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
    max_input_tokens: int = Field(default=4096, ge=1, le=32768)


class BenchmarkRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    iterations: int = Field(default=5, ge=1, le=100)
    warmup_runs: int = Field(default=1, ge=0, le=10)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)
    do_sample: bool = False
    temperature: float = Field(default=0.7, gt=0, le=5)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5)
