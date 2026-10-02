from a11_llm_optimizer.metrics import RequestMetrics


def test_metrics_tracks_adapter_usage_and_latency():
    metrics = RequestMetrics()

    metrics.record_request("Write a Python script.", "python", 120.5)
    metrics.record_request("Explain diabetes symptoms.", "medical", 90.0)

    snapshot = metrics.snapshot()

    assert snapshot["total_requests"] == 2
    assert snapshot["by_adapter"]["python"] == 1
    assert snapshot["by_adapter"]["medical"] == 1
    assert snapshot["avg_latency_ms"] > 0


def test_metrics_retains_bounded_events_without_prompt_text():
    metrics = RequestMetrics(max_events=2)

    metrics.record_request("private prompt one", "python", 10.0)
    metrics.record_request("private prompt two", "medical", 20.0, success=False)
    metrics.record_request("private prompt three", "python", 30.0)

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 3
    assert snapshot["retained_events"] == 2
    assert snapshot["avg_latency_ms"] == 20.0
    assert snapshot["failures"] == 1
    assert len(metrics.events) == 2
    assert all("prompt" not in event for event in metrics.events)
