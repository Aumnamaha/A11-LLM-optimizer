"""Static adapter inventory for scanning known adapter locations and scoring them."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class AdapterInventory:
    """Tracks adapter metadata and simple quality scoring for each known adapter."""

    def __init__(
        self,
        adapter_dir: str = "adapters",
        adapter_map: dict[str, str] | None = None,
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.adapter_map = {key.lower(): value for key, value in (adapter_map or {}).items()}

    def scan(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for name, path in self.adapter_map.items():
            resolved = Path(path)
            exists = resolved.exists() or (self.adapter_dir / resolved.name).exists()
            score = 95 if exists else 40
            summary[name] = {
                "name": name,
                "path": path,
                "exists": exists,
                "score": score,
            }
        return summary
