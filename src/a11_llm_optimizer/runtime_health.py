"""Runtime health checks for local model and adapter directories."""

from __future__ import annotations

from pathlib import Path


class RuntimeHealth:
    """Check whether required local directories are present and usable."""

    def __init__(self, paths: dict[str, str] | None = None) -> None:
        self.paths = {
            "models": "models",
            "adapters": "adapters",
            **(paths or {}),
        }

    def check(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, path in self.paths.items():
            path_obj = Path(path)
            result[name] = "ok" if path_obj.exists() and path_obj.is_dir() else "missing"
        return result
