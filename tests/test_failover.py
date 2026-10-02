from a11_llm_optimizer.failover import FailoverPolicy


def test_failover_uses_default_adapter_when_target_missing():
    policy = FailoverPolicy(default_adapter="adapters/default.gguf")

    route = policy.resolve("unsupported_domain")

    assert route == "adapters/default.gguf"
