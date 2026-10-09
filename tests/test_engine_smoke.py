"""Fast engine smoke tests.

These tests intentionally avoid the 70GB GGUF model shards: they exercise the
pure-Python/numpy paths of the engine so CI stays lightweight and hermetic.
"""
import importlib

import numpy as np
import torch

from engine import quant
from engine.context_manager import WorkspaceContextManager
from engine.expert_cache import HotExpertRAMCache
from engine.quant import GGUFDequantizer
from engine.router import MoERouter
from engine.sampler import TokenSampler


def test_all_engine_modules_import():
    modules = [
        "engine.context_manager",
        "engine.expert_cache",
        "engine.layer_splitter",
        "engine.loader",
        "engine.mmap_loader",
        "engine.prefetcher",
        "engine.quant",
        "engine.router",
        "engine.sampler",
        "engine.speculative",
        "engine.tokenizer",
        "engine.pipeline",
        "engine.server",
    ]
    for module in modules:
        importlib.import_module(module)


def test_router_returns_top_k_with_normalized_weights():
    router = MoERouter(num_experts=16, top_k=4)
    hidden_states = torch.randn(2, 8)
    router_weights = torch.randn(8, 16)

    indices, weights = router.select_experts(hidden_states, router_weights)

    assert indices.shape == (2, 4)
    assert weights.shape == (2, 4)
    assert (weights > 0).all()

    # Selected weights must match the corresponding full-softmax probabilities.
    full_probs = torch.softmax(torch.matmul(hidden_states, router_weights), dim=-1)
    assert torch.allclose(weights, full_probs.gather(-1, indices), atol=1e-6)


def test_expert_cache_lru_hits_and_misses():
    cache = HotExpertRAMCache(max_hot_experts=2)

    assert cache.get_expert("missing") is None
    cache.put_expert("a", b"\x01")
    cache.put_expert("b", b"\x02")
    assert cache.get_expert("a") == b"\x01"

    stats = cache.get_stats()
    assert stats["cached_experts"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0


def test_quant_numpy_fallback_bit_unpacking(monkeypatch):
    monkeypatch.setattr(quant, "HAS_CPP_EXTENSION", False)

    raw = bytes([0xE4]) * 4  # 0b11100100 -> [-1, 0, 1, 2] after 2-bit unpack
    out = GGUFDequantizer.dequantize_q2_k(raw, num_elements=16)

    expected = np.tile(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), 4)
    assert out.shape == (16,)
    np.testing.assert_allclose(out.numpy(), expected)


def test_sampler_returns_valid_token_id():
    sampler = TokenSampler(temperature=1.0, top_k=5, top_p=1.0)
    logits = torch.randn(50)

    token_id = sampler.sample_next_token(logits)

    assert isinstance(token_id, int)
    assert 0 <= token_id < 50


def test_context_manager_builds_workspace_prompt(tmp_path):
    (tmp_path / "sample.py").write_text("print('hi')\n")

    manager = WorkspaceContextManager(str(tmp_path))
    prompt = manager.build_workspace_prompt("explain this")

    assert "explain this" in prompt
    assert "sample.py" in prompt
