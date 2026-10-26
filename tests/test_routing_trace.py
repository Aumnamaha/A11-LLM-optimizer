"""Hermetic tests for the routing-locality statistics (no model required)."""
from tools.model_analyzer.routing_trace import compute_stats


def _trace():
    return [
        {"prompt": 0, "token": 0, "layer": 0, "experts": [0, 1]},
        {"prompt": 0, "token": 0, "layer": 1, "experts": [0, 1]},
        {"prompt": 0, "token": 1, "layer": 0, "experts": [0, 1]},
        {"prompt": 0, "token": 1, "layer": 1, "experts": [1, 2]},
    ]


def test_compute_stats_counts_and_uniques():
    stats = compute_stats(_trace(), num_layers=2)
    assert stats["selections"] == 8
    assert stats["tokens_traced"] == 2
    assert stats["unique_experts_all_layers"] == 3
    assert stats["unique_layer_expert_pairs"] == 5
    assert stats["per_layer_unique_experts"] == {"0": 2, "1": 3}


def test_compute_stats_locality_jaccards():
    stats = compute_stats(_trace(), num_layers=2)
    # consecutive-layer: (2/2) for token 0 and (1/3) for token 1 -> 3/5
    assert stats["consecutive_layer_jaccard"] == 0.6
    # temporal: layer 0 identical, layer 1 shares {1} -> 3/5
    assert stats["temporal_token_jaccard"] == 0.6


def test_compute_stats_hot_set_coverage():
    stats = compute_stats(_trace(), num_layers=2)
    # counts sorted desc: [2,2,2,1,1], total 8
    # cumulative: 2,4,6,7,8 -> 90% (7.2) is first reached at 5 experts
    assert stats["hot_set_50pct"] == 2
    assert stats["hot_set_90pct"] == 5


def test_compute_stats_handles_empty_trace():
    stats = compute_stats([], num_layers=2)
    assert stats["selections"] == 0
    assert stats["consecutive_layer_jaccard"] == 0.0
    assert stats["temporal_token_jaccard"] == 0.0
