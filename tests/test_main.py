import main


def test_standalone_entrypoint_uses_shared_completion_client(monkeypatch, capsys):
    captured = {}

    class FakeClient:
        base_url = "http://localhost:8081"

        def __init__(self, **kwargs):
            pass

        def completion(self, payload):
            captured.update(payload)
            return {
                "content": "Generated from the backend",
                "timings": {"prompt_n": 5, "prompt_ms": 10, "prompt_per_second": 500.0},
            }

    monkeypatch.setattr(main, "LocalLLMClient", FakeClient)

    main.run_inference("Write a Python script.")

    output = capsys.readouterr().out
    assert captured["adapter"] == "python"
    assert "Generated from the backend" in output
    assert "500.00 tokens/s" in output
