"""Local model backend abstraction for adapter-aware generation requests."""

from __future__ import annotations

from typing import Any

from .local_client import LocalLLMClient


class ModelBackend:
    """Convenience wrapper for routed calls to a local llama.cpp server."""

    def __init__(self, base_url: str | None = None) -> None:
        self.client = LocalLLMClient(base_url=base_url)

    def build_payload(
        self,
        prompt: str,
        adapter: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "adapter": adapter,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }

    def generate(
        self,
        prompt: str,
        adapter: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        payload = self.build_payload(
            prompt=prompt,
            adapter=adapter,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self.client.completion(payload)
