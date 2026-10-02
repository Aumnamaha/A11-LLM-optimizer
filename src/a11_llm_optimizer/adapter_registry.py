"""Registry for adapter metadata and selection."""

from __future__ import annotations


class AdapterRegistry:
    """Simple mapping from intent names to adapter file paths."""

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        default_path: str = "adapters/default.gguf",
    ) -> None:
        self.mapping = {key.lower(): value for key, value in (mapping or {}).items()}
        self.default_path = default_path

    def get(self, adapter_name: str) -> str:
        """Return the adapter path for a named intent or the default adapter."""
        normalized = (adapter_name or "").strip().lower()
        return self.mapping.get(normalized, self.default_path)

    def list_adapters(self) -> dict[str, str]:
        return dict(self.mapping)
