"""FastAPI interface for the optimizer router and local generation flow."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .local_client import LocalLLMClient
from .optimizer import Optimizer


class RouteRequest(BaseModel):
    prompt: str
    max_tokens: int = 256


class GenerateRequest(RouteRequest):
    temperature: float = 0.3


app = FastAPI(title="A11 LLM Optimizer")
optimizer = Optimizer()
client = LocalLLMClient()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/route")
def route(request: RouteRequest) -> dict:
    result = optimizer.optimize(request.prompt, max_tokens=request.max_tokens)
    return {
        "adapter": result["adapter"],
        "adapter_path": result["adapter_path"],
        "request": result["request"],
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    result = optimizer.optimize(request.prompt, max_tokens=request.max_tokens)
    response = client.completion(result["request"])
    return {
        "adapter": result["adapter"],
        "adapter_path": result["adapter_path"],
        "response": response,
        "request": result["request"],
    }
