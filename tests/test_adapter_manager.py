from a11_llm_optimizer.adapter_manager import AdapterManager


def test_loads_valid_adapter_when_present():
    manager = AdapterManager(
        {
            "python": "adapters/python_coder.gguf",
            "medical": "adapters/medical_lora.gguf",
        }
    )

    assert manager.load("python") == "adapters/python_coder.gguf"
    assert manager.is_available("python") is True


def test_falls_back_to_default_when_missing():
    manager = AdapterManager(
        {
            "python": "adapters/python_coder.gguf",
        }
    )

    assert manager.load("creative") == "adapters/default.gguf"
    assert manager.is_available("creative") is False
