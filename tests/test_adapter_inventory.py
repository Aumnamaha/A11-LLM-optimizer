from a11_llm_optimizer.adapter_inventory import AdapterInventory


def test_inventory_discovers_adapters_and_scores_them():
    inventory = AdapterInventory(
        adapter_dir="adapters",
        adapter_map={
            "python": "adapters/python_coder.gguf",
            "medical": "adapters/medical_lora.gguf",
        },
    )

    summary = inventory.scan()

    assert "python" in summary
    assert "medical" in summary
    assert summary["python"]["path"] == "adapters/python_coder.gguf"
    assert summary["python"]["score"] >= 0
