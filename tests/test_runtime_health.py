from a11_llm_optimizer.runtime_health import RuntimeHealth


def test_runtime_health_reports_directory_state():
    health = RuntimeHealth(paths={"models": "models", "adapters": "adapters"})
    status = health.check()

    assert status["models"] in {"ok", "missing"}
    assert status["adapters"] in {"ok", "missing"}
