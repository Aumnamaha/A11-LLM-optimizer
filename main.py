from typing import Any

import httpx

from a11_llm_optimizer.adapter_registry import AdapterRegistry
from a11_llm_optimizer.config import settings
from a11_llm_optimizer.local_client import LocalLLMClient
from a11_llm_optimizer.optimizer import Optimizer


def run_inference(prompt: str) -> None:
    optimizer = Optimizer(
        registry=AdapterRegistry(settings.adapter_paths, settings.default_adapter_path)
    )
    client = LocalLLMClient(adapter_ids=settings.adapter_ids)

    print(f"\n[1] Evaluating Prompt: {prompt}")
    result = optimizer.optimize(prompt)
    print(f"[2] Selected adapter: {result['adapter']} ({result['adapter_path']})")

    try:
        response = client.completion(result["request"])
    except httpx.HTTPError as error:
        print(f"Inference backend request failed at {client.base_url}: {error}")
        return

    print("\n--- Output ---")
    print(str(response.get("content", "")).strip())
    timings: dict[str, Any] = response.get("timings", {})
    if timings:
        print("\n--- Timing ---")
        for label, prefix in (("Prompt", "prompt"), ("Generated", "predicted")):
            count = timings.get(f"{prefix}_n") or 0
            duration = timings.get(f"{prefix}_ms") or 0
            rate = timings.get(f"{prefix}_per_second") or 0.0
            print(f"{label}: {count} tokens in {duration} ms ({rate:.2f} tokens/s)")


if __name__ == "__main__":
    run_inference("Write a Python script to scrape a website.")
