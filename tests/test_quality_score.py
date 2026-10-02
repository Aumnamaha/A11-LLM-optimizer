from a11_llm_optimizer.quality_score import QualityScorer


def test_quality_scorer_returns_numeric_scores():
    scorer = QualityScorer()

    score = scorer.score(
        prompt="Write a Python API endpoint.",
        adapter="python",
        latency_ms=120,
        success=True,
    )

    assert 0 <= score <= 100
