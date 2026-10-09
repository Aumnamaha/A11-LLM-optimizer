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
        "engine.hardware",
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


def test_detect_hardware_returns_valid_profile():
    from engine.hardware import detect_hardware, HardwareProfile

    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb > 0.0
    assert profile.available_ram_gb > 0.0
    assert profile.cpu_count > 0
    assert isinstance(profile.gpu_name, str)
    assert profile.total_vram_gb >= 0.0
    assert profile.free_vram_gb >= 0.0


def test_evaluate_compatibility_clamps_over_allocated_budgets():
    from engine.hardware import (
        HardwareProfile,
        ModelHardwareRequirements,
        evaluate_compatibility,
    )

    profile = HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        used_ram_gb=8.0,
        ram_utilization_pct=50.0,
        gpu_detected=True,
        gpu_name="Test GPU",
        gpu_backend="test",
        total_vram_gb=4.0,
        free_vram_gb=4.0,
        has_discrete_gpu=True,
        cpu_count=8,
        platform_name="Linux",
    )
    req = ModelHardwareRequirements(
        model_name="Test MoE",
        total_model_size_gb=18.0,
        core_heads_size_gb=1.0,
        moe_experts_size_gb=17.0,
        ngram_size_gb=0.0,
        min_ram_required_gb=2.5,
        recommended_ram_gb=6.0,
        recommended_vram_gb=2.0,
    )

    # Request excessive budgets that exceed hardware
    report = evaluate_compatibility(profile, req, requested_vram_gb=12.0, requested_ram_gb=32.0)
    assert report.is_compatible is True
    assert report.effective_vram_budget_gb <= profile.total_vram_gb
    assert report.effective_ram_budget_gb <= profile.total_ram_gb
    assert len(report.warnings) >= 2


def test_evaluate_compatibility_cpu_fallback_mode():
    from engine.hardware import (
        HardwareProfile,
        ModelHardwareRequirements,
        evaluate_compatibility,
    )

    profile = HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        used_ram_gb=6.0,
        ram_utilization_pct=37.5,
        gpu_detected=False,
        gpu_name="None (CPU)",
        gpu_backend="cpu",
        total_vram_gb=0.0,
        free_vram_gb=0.0,
        has_discrete_gpu=False,
        cpu_count=8,
        platform_name="Linux",
    )
    req = ModelHardwareRequirements(
        model_name="Test MoE",
        total_model_size_gb=18.0,
        core_heads_size_gb=1.0,
        moe_experts_size_gb=17.0,
        ngram_size_gb=0.0,
        min_ram_required_gb=2.5,
        recommended_ram_gb=6.0,
        recommended_vram_gb=2.0,
    )

    report = evaluate_compatibility(profile, req)
    assert report.is_compatible is True
    assert report.effective_vram_budget_gb == 0.0
    assert report.effective_ram_budget_gb > 0.0
    assert "CPU" in report.offload_strategy


def test_layer_splitter_cpu_mode_and_budgets():
    from unittest.mock import MagicMock
    from engine.layer_splitter import LayerMemorySplitter

    mock_loader = MagicMock()
    # 1 GB core tensor, 2 GB expert tensor, 1 GB cold expert
    mock_loader.tensors = {
        "blk.0.attn_q.weight": {"n_bytes": 1024 ** 3},
        "blk.0.ffn_down_exps.weight": {"n_bytes": 2 * (1024 ** 3)},
        "blk.1.ffn_down_exps.weight": {"n_bytes": 2 * (1024 ** 3)},
    }

    # Case 1: CPU mode (vram=0, ram=4GB)
    splitter = LayerMemorySplitter(mock_loader, vram_budget_gb=0.0, ram_budget_gb=4.0)
    assert len(splitter.vram_tensors) == 0
    # Core tensor falls back to RAM
    assert "blk.0.attn_q.weight" in splitter.ram_tensors
    # One expert fits in remaining RAM (1GB + 2GB = 3GB <= 4GB)
    assert "blk.0.ffn_down_exps.weight" in splitter.ram_tensors
    # Other expert goes to SSD
    assert "blk.1.ffn_down_exps.weight" in splitter.ssd_tensors

