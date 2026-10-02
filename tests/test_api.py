from fastapi.testclient import TestClient

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


def test_generate_endpoint_returns_request_payload():
    response = client.post(
        "/generate",
        json={"prompt": "Explain diabetic symptoms and treatment.", "max_tokens": 128},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["adapter"] == "medical"
    assert payload["request"]["max_tokens"] == 128
