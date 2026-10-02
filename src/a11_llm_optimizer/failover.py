"""Failover logic for missing or unsupported adapters."""

from __future__ import annotations


class FailoverPolicy:
    """Choose a safe fallback adapter when the preferred route is not available."""

    def __init__(
        self,
        default_adapter: str = "adapters/default.gguf",
        adapter_map: dict[str, str] | None = None,
    ) -> None:
        self.default_adapter = default_adapter
        self.adapter_map = {
            "python": "adapters/python_coder.gguf",
            "medical": "adapters/medical_lora.gguf",
            "creative": "adapters/creative_writer.gguf",
            **(adapter_map or {}),
        }

    def resolve(self, adapter_name: str | None) -> str:
        name = (adapter_name or "").strip().lower()
        if not name:
            return self.default_adapter
        if name in self.adapter_map:
            return self.adapter_map[name]
        return self.default_adapter
