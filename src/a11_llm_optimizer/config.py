"""Application configuration for local runtime settings."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model_base_path: str = os.getenv("MODEL_BASE_PATH", "./models")
    base_model_file: str = os.getenv("BASE_MODEL_FILE", "base-3b-q4_k_m.gguf")
    adapters_path: str = os.getenv("ADAPTERS_PATH", "./adapters")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    default_adapter_path: str = os.getenv("DEFAULT_ADAPTER_PATH", "")
    llama_server_url: str = os.getenv("LLAMA_SERVER_URL", "http://localhost:8081")
    llama_server_timeout_seconds: float = float(os.getenv("LLAMA_SERVER_TIMEOUT_SECONDS", "120"))
    llama_server_host: str = os.getenv("LLAMA_SERVER_HOST", "0.0.0.0")
    llama_server_port: int = int(os.getenv("LLAMA_SERVER_PORT", "8081"))
    llama_context_size: int = int(os.getenv("LLAMA_CONTEXT_SIZE", "2048"))
    llama_gpu_layers: str = os.getenv("LLAMA_GPU_LAYERS", "auto")
    llama_prompt_format: str = os.getenv("LLAMA_PROMPT_FORMAT", "qwen-chatml")
    python_adapter_file: str = os.getenv("PYTHON_ADAPTER_FILE", "python_coder.gguf")
    medical_adapter_file: str = os.getenv("MEDICAL_ADAPTER_FILE", "medical_expert.gguf")
    creative_adapter_file: str = os.getenv("CREATIVE_ADAPTER_FILE", "creative_writer.gguf")

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.llama_server_port <= 65535:
            raise ValueError("llama_server_port must be between 1 and 65535")
        if (
            not math.isfinite(self.llama_server_timeout_seconds)
            or self.llama_server_timeout_seconds <= 0
        ):
            raise ValueError("llama_server_timeout_seconds must be positive and finite")
        if self.llama_context_size < 1:
            raise ValueError("llama_context_size must be positive")
        try:
            numeric_gpu_layers = int(self.llama_gpu_layers)
        except ValueError as error:
            if self.llama_gpu_layers not in {"auto", "all"}:
                raise ValueError(
                    "llama_gpu_layers must be auto, all, or an integer >= -1"
                ) from error
        else:
            if numeric_gpu_layers < -1:
                raise ValueError("llama_gpu_layers must be auto, all, or an integer >= -1")
        if self.llama_prompt_format not in {"qwen-chatml", "raw"}:
            raise ValueError("llama_prompt_format must be 'qwen-chatml' or 'raw'")
        if not self.default_adapter_path:
            self.default_adapter_path = str(Path(self.adapters_path) / "default.gguf")

    @property
    def model_path(self) -> str:
        return str(Path(self.model_base_path) / self.base_model_file)

    @property
    def adapter_paths(self) -> dict[str, str]:
        adapter_dir = Path(self.adapters_path)
        return {
            "python": str(adapter_dir / self.python_adapter_file),
            "medical": str(adapter_dir / self.medical_adapter_file),
            "creative": str(adapter_dir / self.creative_adapter_file),
        }

    @property
    def adapter_ids(self) -> dict[str, int]:
        return {"python": 0, "medical": 1, "creative": 2}


settings = Settings()
