from a11_llm_optimizer.router import IntentRouter


def test_matches_python_intent():
    router = IntentRouter()

    route = router.route("Write a Python script to scrape a website.")

    assert route == "python"


def test_matches_medical_intent():
    router = IntentRouter()

    route = router.route("Explain the symptoms and treatment plan for diabetes.")

    assert route == "medical"


def test_builds_request_with_adapter_scale():
    router = IntentRouter()

    payload = router.build_request("Write a Python API endpoint.", max_tokens=256)

    assert payload["prompt"] == "Write a Python API endpoint."
    assert payload["max_tokens"] == 256
    assert payload["adapter"] == "python"
    assert payload["scale"] == 1.0
