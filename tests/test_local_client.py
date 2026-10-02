from typing import Any

from a11_llm_optimizer import local_client
from a11_llm_optimizer.local_client import LocalLLMClient


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, str]:
        return {"content": "Generated answer"}


def test_completion_posts_llama_cpp_payload(monkeypatch):
    call: dict[str, Any] = {}

    def fake_post(url, *, json, timeout):
        call.update(url=url, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(local_client.httpx, "post", fake_post)
    backend = LocalLLMClient(
        base_url="http://localhost:8081/",
        timeout_seconds=30,
        adapter_ids={"python": 0, "medical": 1},
    )

    result = backend.completion(
        {
            "prompt": "Write Python code.",
            "adapter": "python",
            "max_tokens": 100,
            "temperature": 0.4,
        }
    )

    assert call["url"] == "http://localhost:8081/completion"
    assert call["timeout"] == 30
    assert call["json"] == {
        "prompt": "<|im_start|>user\nWrite Python code.<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": 100,
        "temperature": 0.4,
        "lora": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}],
    }
    assert result == {"content": "Generated answer"}


def test_completion_disables_all_adapters_for_general_prompts(monkeypatch):
    call: dict[str, Any] = {}

    def fake_post(url, *, json, timeout):
        call.update(json=json)
        return FakeResponse()

    monkeypatch.setattr(local_client.httpx, "post", fake_post)
    backend = LocalLLMClient(
        adapter_ids={"python": 0, "medical": 1, "creative": 2}, prompt_format="raw"
    )

    backend.completion({"prompt": "Hello", "adapter": "general"})

    assert call["json"]["lora"] == [
        {"id": 0, "scale": 0.0},
        {"id": 1, "scale": 0.0},
        {"id": 2, "scale": 0.0},
    ]
    assert call["json"]["prompt"] == "Hello"
