import httpx
from fastapi.testclient import TestClient

from a11_llm_optimizer import api
from a11_llm_optimizer.api import app

client = TestClient(app)


def test_route_endpoint_returns_adapter_choice():
    response = client.post(
        "/route",
        json={"prompt": "Write a Python script to scrape a website."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["adapter"] == "python"
    assert payload["adapter_path"] == "adapters/python_coder.gguf"


def test_generate_endpoint_passes_parameters_and_returns_backend_response(monkeypatch):
    captured_payload = {}

    def fake_completion(payload):
        captured_payload.update(payload)
        return {"content": "Generated answer"}

    monkeypatch.setattr(api.client, "completion", fake_completion)
    response = client.post(
        "/generate",
        json={
            "prompt": "Explain diabetic symptoms and treatment.",
            "max_tokens": 128,
            "temperature": 0.6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["adapter"] == "medical"
    assert payload["request"]["max_tokens"] == 128
    assert payload["request"]["temperature"] == 0.6
    assert captured_payload["temperature"] == 0.6
    assert payload["response"] == {"content": "Generated answer"}


def test_generate_endpoint_reports_backend_failure(monkeypatch):
    def fail_completion(payload):
        request = httpx.Request("POST", "http://localhost:8081/completion")
        raise httpx.ConnectError("backend unavailable", request=request)

    monkeypatch.setattr(api.client, "completion", fail_completion)
    response = client.post("/generate", json={"prompt": "Write Python code."})

    assert response.status_code == 502
    assert response.json()["detail"] == "Inference backend request failed"


def test_generate_endpoint_rejects_invalid_generation_parameters():
    response = client.post(
        "/generate",
        json={"prompt": "Write Python code.", "max_tokens": 0, "temperature": 3.0},
    )

    assert response.status_code == 422


def test_route_endpoint_rejects_blank_prompt():
    response = client.post("/route", json={"prompt": ""})

    assert response.status_code == 422


def test_ready_endpoint_returns_service_unavailable_until_ready(monkeypatch):
    monkeypatch.setattr(api.runtime_health, "check", lambda: {"base_model": "missing"})

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"


def test_ready_endpoint_returns_success_when_all_checks_pass(monkeypatch):
    monkeypatch.setattr(
        api.runtime_health,
        "check",
        lambda: {"base_model": "ok", "inference_backend": "ok"},
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
