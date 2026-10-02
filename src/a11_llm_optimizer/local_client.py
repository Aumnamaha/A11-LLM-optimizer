"""Thin wrapper around local model-server requests."""

from __future__ import annotations

from typing import Any


class LocalLLMClient:
    """Placeholder client for a local llama.cpp or similar inference endpoint."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")

    def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a simulated response object for local development."""
        return {
            "status": "ok",
            "base_url": self.base_url,
            "adapter": payload.get("adapter", "general"),
            "request": payload,
        }
