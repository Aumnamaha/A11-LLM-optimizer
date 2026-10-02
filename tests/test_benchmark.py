from a11_llm_optimizer.benchmark import BenchmarkRunner


def test_benchmark_runner_returns_domain_results():
    runner = BenchmarkRunner(
        prompts={
            "python": ["Write a Python script to scrape a website."],
            "medical": ["Explain symptoms and treatment for diabetes."],
        }
    )

    results = runner.run()

    assert "python" in results
    assert "medical" in results
    assert results["python"][0]["domain"] == "python"
    assert results["medical"][0]["domain"] == "medical"
