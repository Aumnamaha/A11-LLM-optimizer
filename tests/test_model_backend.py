from a11_llm_optimizer.model_backend import ModelBackend


def test_backend_builds_local_request_payload():
    backend = ModelBackend(base_url="http://localhost:8080")

    payload = backend.build_payload(
        prompt="Write a Python script for a CLI tool.",
        adapter="python",
        max_tokens=128,
        temperature=0.2,
    )

    assert payload["prompt"] == "Write a Python script for a CLI tool."
    assert payload["adapter"] == "python"
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.2


def test_backend_returns_status_for_simulated_generation():
    backend = ModelBackend(base_url="http://localhost:8080")

    result = backend.generate(
        prompt="Explain medical symptoms in simple terms.",
        adapter="medical",
        max_tokens=64,
    )

    assert result["status"] == "ok"
    assert result["adapter"] == "medical"
