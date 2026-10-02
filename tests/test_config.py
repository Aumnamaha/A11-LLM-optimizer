from a11_llm_optimizer.config import settings


def test_default_settings_are_loaded():
    assert settings.model_base_path == "./models"
    assert settings.adapters_path == "./adapters"
    assert settings.port == 8080
