"""Thin wrapper around local model-server requests."""

from __future__ import annotations

from typing import Any

import httpx

from .config import settings


class LocalLLMClient:
    """Client for a local llama.cpp server with preloaded LoRA adapters."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        adapter_ids: dict[str, int] | None = None,
        prompt_format: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llama_server_url).rstrip("/")
        self.timeout_seconds = (
            settings.llama_server_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        self.adapter_ids = settings.adapter_ids if adapter_ids is None else adapter_ids
        self.prompt_format = (
            settings.llama_prompt_format if prompt_format is None else prompt_format
        )

    def _format_prompt(self, prompt: str) -> str:
        if self.prompt_format == "raw":
            return prompt
        if self.prompt_format == "qwen-chatml":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        raise ValueError(f"Unsupported prompt format: {self.prompt_format}")

    def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate a completion using llama.cpp's server API."""
        selected_adapter = str(payload.get("adapter", "general")).lower()
        selected_adapter_id = self.adapter_ids.get(selected_adapter)
        request_payload = {
            "prompt": self._format_prompt(str(payload["prompt"])),
            "n_predict": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.3)),
            "lora": [
                {
                    "id": adapter_id,
                    "scale": float(payload.get("scale", 1.0))
                    if adapter_id == selected_adapter_id
                    else 0.0,
                }
                for adapter_id in self.adapter_ids.values()
            ],
        }
        response = httpx.post(
            f"{self.base_url}/completion",
            json=request_payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
