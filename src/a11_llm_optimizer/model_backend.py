"""Local model backend abstraction for adapter-aware generation requests."""

from __future__ import annotations

from typing import Any


class ModelBackend:
    """Simple wrapper for interacting with a local llama.cpp or compatible backend."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")

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
        return {
            "status": "ok",
            "base_url": self.base_url,
            "adapter": adapter,
            "payload": payload,
        }
