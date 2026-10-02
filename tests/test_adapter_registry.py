from a11_llm_optimizer.adapter_registry import AdapterRegistry


def test_get_known_adapter():
    registry = AdapterRegistry(
        {
            "python": "adapters/python_coder.gguf",
            "medical": "adapters/medical_lora.gguf",
        }
    )

    assert registry.get("python") == "adapters/python_coder.gguf"
    assert registry.get("medical") == "adapters/medical_lora.gguf"


def test_get_unknown_adapter_falls_back_to_default():
    registry = AdapterRegistry({"python": "adapters/python_coder.gguf"})

    assert registry.get("creative") == "adapters/default.gguf"
