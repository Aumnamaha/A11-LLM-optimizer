"""Adapter loading and fallback management for domain-specific routing."""

from __future__ import annotations


class AdapterManager:
    """Tracks adapter paths and resolves default fallback behavior."""

    def __init__(
        self,
        adapters: dict[str, str] | None = None,
        default_path: str = "adapters/default.gguf",
    ) -> None:
        self.adapters = {key.lower(): value for key, value in (adapters or {}).items()}
        self.default_path = default_path

    def load(self, name: str) -> str:
        """Return a valid adapter path or the default adapter if not available."""
        normalized = (name or "").strip().lower()
        if normalized in self.adapters:
            return self.adapters[normalized]
        return self.default_path

    def is_available(self, name: str) -> bool:
        """Check whether the requested adapter exists in the registry."""
        return (name or "").strip().lower() in self.adapters
