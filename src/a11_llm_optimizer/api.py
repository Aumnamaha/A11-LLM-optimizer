"""FastAPI interface for the optimizer router and local generation flow."""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

from .adapter_inventory import AdapterInventory
from .local_client import LocalLLMClient
from .metrics import RequestMetrics
from .optimizer import Optimizer
from .quality_score import QualityScorer


class RouteRequest(BaseModel):
    prompt: str
    max_tokens: int = 256


class GenerateRequest(RouteRequest):
    temperature: float = 0.3


app = FastAPI(title="A11 LLM Optimizer")
optimizer = Optimizer()
client = LocalLLMClient()
metrics = RequestMetrics()
inventory = AdapterInventory(
    adapter_map={
        "python": "adapters/python_coder.gguf",
        "medical": "adapters/medical_lora.gguf",
        "creative": "adapters/creative_writer.gguf",
    }
)
scorer = QualityScorer()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "metrics": metrics.snapshot(),
        "inventory": inventory.scan(),
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.get("/inventory")
def get_inventory() -> dict:
    return inventory.scan()


@app.post("/route")
def route(request: RouteRequest) -> dict:
    start = time.perf_counter()
    result = optimizer.optimize(request.prompt, max_tokens=request.max_tokens)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record_request(request.prompt, result["adapter"], latency_ms)
    score = scorer.score(
        prompt=request.prompt,
        adapter=result["adapter"],
        latency_ms=latency_ms,
        success=True,
    )
    return {
        "adapter": result["adapter"],
        "adapter_path": result["adapter_path"],
        "request": result["request"],
        "latency_ms": round(latency_ms, 3),
        "quality_score": score,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    start = time.perf_counter()
    result = optimizer.optimize(request.prompt, max_tokens=request.max_tokens)
    response = client.completion(result["request"])
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record_request(request.prompt, result["adapter"], latency_ms)
    score = scorer.score(
        prompt=request.prompt,
        adapter=result["adapter"],
        latency_ms=latency_ms,
        success=True,
    )
    return {
        "adapter": result["adapter"],
        "adapter_path": result["adapter_path"],
        "response": response,
        "request": result["request"],
        "latency_ms": round(latency_ms, 3),
        "quality_score": score,
    }
