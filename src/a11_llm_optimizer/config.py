"""Application configuration for local runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Settings:
    model_base_path: str = os.getenv("MODEL_BASE_PATH", "./models")
    adapters_path: str = os.getenv("ADAPTERS_PATH", "./adapters")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    default_adapter_path: str = os.getenv("DEFAULT_ADAPTER_PATH", "./adapters/default.gguf")


settings = Settings()
