"""FastAPI interface for the optimizer router and local generation flow."""

from __future__ import annotations

import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .adapter_inventory import AdapterInventory
from .adapter_registry import AdapterRegistry
from .config import settings
from .failover import FailoverPolicy
from .local_client import LocalLLMClient
from .metrics import RequestMetrics
from .optimizer import Optimizer
from .quality_score import QualityScorer
from .runtime_health import RuntimeHealth


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32768)
    max_tokens: int = Field(default=256, ge=1, le=8192)


class GenerateRequest(RouteRequest):
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


app = FastAPI(title="A11 LLM Optimizer")
optimizer = Optimizer(
    registry=AdapterRegistry(settings.adapter_paths, settings.default_adapter_path)
)
client = LocalLLMClient(adapter_ids=settings.adapter_ids)
metrics = RequestMetrics()
inventory = AdapterInventory(
    adapter_dir=settings.adapters_path,
    adapter_map=settings.adapter_paths,
)
scorer = QualityScorer()
failover = FailoverPolicy(
    default_adapter=settings.default_adapter_path,
    adapter_map=settings.adapter_paths,
)
runtime_health = RuntimeHealth(
    paths={"models": settings.model_base_path, "adapters": settings.adapters_path},
    required_files={"base_model": settings.model_path, **settings.adapter_paths},
    backend_url=settings.llama_server_url,
    timeout_seconds=min(settings.llama_server_timeout_seconds, 2.0),
)


@app.get("/health")
def health() -> dict:
    result = runtime_health.check()
    return {
        "status": "ok" if all(v == "ok" for v in result.values()) else "warning",
        "metrics": metrics.snapshot(),
        "inventory": inventory.scan(),
        "runtime": result,
    }


@app.get("/ready")
def ready() -> dict:
    result = runtime_health.check()
    if not all(status == "ok" for status in result.values()):
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": result},
        )
    return {"status": "ready", "checks": result}


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
    adapter_name = result["adapter"]
    selected_adapter = failover.resolve(adapter_name)
    result["adapter_path"] = selected_adapter
    latency_ms = (time.perf_counter() - start) * 1000
    metrics.record_request(request.prompt, adapter_name, latency_ms)
    score = scorer.score(
        prompt=request.prompt,
        adapter=adapter_name,
        latency_ms=latency_ms,
        success=True,
    )
    return {
        "adapter": adapter_name,
        "adapter_path": selected_adapter,
        "request": result["request"],
        "latency_ms": round(latency_ms, 3),
        "quality_score": score,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    start = time.perf_counter()
    result = optimizer.optimize(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    try:
        response = client.completion(result["request"])
    except httpx.HTTPError as error:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record_request(request.prompt, result["adapter"], latency_ms, success=False)
        raise HTTPException(status_code=502, detail="Inference backend request failed") from error
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
