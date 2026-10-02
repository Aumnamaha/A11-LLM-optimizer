import httpx

from a11_llm_optimizer import runtime_health
from a11_llm_optimizer.runtime_health import RuntimeHealth


def test_runtime_health_reports_directory_state():
    health = RuntimeHealth(paths={"models": "models", "adapters": "adapters"})
    status = health.check()

    assert status["models"] in {"ok", "missing"}
    assert status["adapters"] in {"ok", "missing"}


def test_runtime_health_checks_model_files(tmp_path):
    model_file = tmp_path / "base.gguf"
    health = RuntimeHealth(required_files={"base_model": str(model_file)})

    assert health.check()["base_model"] == "missing"
    model_file.touch()
    assert health.check()["base_model"] == "ok"


def test_runtime_health_checks_backend(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

    call: dict[str, str | float] = {}

    def fake_get(url, *, timeout):
        call.update(url=url, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(runtime_health.httpx, "get", fake_get)
    health = RuntimeHealth(backend_url="http://localhost:8081", timeout_seconds=0.5)

    assert health.check()["inference_backend"] == "ok"
    assert call == {"url": "http://localhost:8081/health", "timeout": 0.5}


def test_runtime_health_reports_unavailable_backend(monkeypatch):
    def fail_get(url, *, timeout):
        request = httpx.Request("GET", url)
        raise httpx.ConnectError("backend unavailable", request=request)

    monkeypatch.setattr(runtime_health.httpx, "get", fail_get)
    health = RuntimeHealth(backend_url="http://localhost:8081")

    assert health.check()["inference_backend"] == "unavailable"
