"""High-level optimizer orchestration."""

from __future__ import annotations

from .adapter_registry import AdapterRegistry
from .router import IntentRouter


class Optimizer:
    """Coordinates routing and adapter loading for a request."""

    def __init__(
        self,
        router: IntentRouter | None = None,
        registry: AdapterRegistry | None = None,
    ) -> None:
        self.router = router or IntentRouter()
        self.registry = registry or AdapterRegistry(
            {
                "python": "adapters/python_coder.gguf",
                "medical": "adapters/medical_lora.gguf",
                "creative": "adapters/creative_writer.gguf",
            }
        )

    def optimize(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> dict:
        adapter_name = self.router.route(prompt)
        request = self.router.build_request(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            adapter=adapter_name,
        )
        return {
            "adapter": adapter_name,
            "adapter_path": self.registry.get(adapter_name),
            "request": request,
        }
