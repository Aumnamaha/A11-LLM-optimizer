from a11_llm_optimizer import local_client
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


def test_backend_returns_real_completion(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": "Generated answer"}

    monkeypatch.setattr(local_client.httpx, "post", lambda *args, **kwargs: FakeResponse())
    backend = ModelBackend(base_url="http://localhost:8081")

    result = backend.generate(
        prompt="Explain medical symptoms in simple terms.",
        adapter="medical",
        max_tokens=64,
    )

    assert result["content"] == "Generated answer"
