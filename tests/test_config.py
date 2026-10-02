import pytest

from a11_llm_optimizer.config import Settings, settings


def test_default_settings_are_loaded():
    assert settings.model_base_path == "./models"
    assert settings.adapters_path == "./adapters"
    assert settings.port == 8080
    assert settings.llama_gpu_layers == "auto"


def test_settings_derive_model_and_adapter_paths(tmp_path):
    config = Settings(
        model_base_path=str(tmp_path / "models"),
        base_model_file="small.gguf",
        adapters_path=str(tmp_path / "adapters"),
        python_adapter_file="code.gguf",
    )

    assert config.model_path == str(tmp_path / "models" / "small.gguf")
    assert config.adapter_paths["python"] == str(tmp_path / "adapters" / "code.gguf")
    assert config.default_adapter_path == str(tmp_path / "adapters" / "default.gguf")
    assert config.adapter_ids == {"python": 0, "medical": 1, "creative": 2}


def test_settings_honor_explicit_default_adapter_path(tmp_path):
    override = tmp_path / "fallback.gguf"
    config = Settings(adapters_path=str(tmp_path / "adapters"), default_adapter_path=str(override))

    assert config.default_adapter_path == str(override)


def test_settings_reject_invalid_runtime_values():
    with pytest.raises(ValueError, match="port"):
        Settings(port=0)
    with pytest.raises(ValueError, match="timeout"):
        Settings(llama_server_timeout_seconds=0)
    with pytest.raises(ValueError, match="context"):
        Settings(llama_context_size=0)
    with pytest.raises(ValueError, match="gpu_layers"):
        Settings(llama_gpu_layers="-2")
    with pytest.raises(ValueError, match="gpu_layers"):
        Settings(llama_gpu_layers="invalid")
    with pytest.raises(ValueError, match="prompt_format"):
        Settings(llama_prompt_format="unsupported")
