"""Runtime health checks for local model and adapter directories."""

from __future__ import annotations

from pathlib import Path

import httpx


class RuntimeHealth:
    """Check local model assets and reachability of the inference backend."""

    def __init__(
        self,
        paths: dict[str, str] | None = None,
        required_files: dict[str, str] | None = None,
        backend_url: str | None = None,
        timeout_seconds: float = 1.0,
    ) -> None:
        self.paths = {
            "models": "models",
            "adapters": "adapters",
            **(paths or {}),
        }
        self.required_files = required_files or {}
        self.backend_url = backend_url.rstrip("/") if backend_url else None
        self.timeout_seconds = timeout_seconds

    def check(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, path in self.paths.items():
            path_obj = Path(path)
            result[name] = "ok" if path_obj.exists() and path_obj.is_dir() else "missing"
        for name, path in self.required_files.items():
            result[name] = "ok" if Path(path).is_file() else "missing"
        if self.backend_url:
            try:
                response = httpx.get(f"{self.backend_url}/health", timeout=self.timeout_seconds)
                response.raise_for_status()
                result["inference_backend"] = "ok"
            except httpx.HTTPError:
                result["inference_backend"] = "unavailable"
        return result
