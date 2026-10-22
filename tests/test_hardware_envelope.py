"""Tests for the runtime's minimum supported hardware envelope.

These verify that budgets are *derived* from the model and hardware rather than
hard-coded, that 2 GB VRAM / 8 GB RAM act as the declared minimum requirement,
and that machines below it are flagged (but not fatal).
"""
from types import SimpleNamespace

from engine.hardware import (
    MIN_RAM_GB,
    MIN_VRAM_GB,
    HardwareProfile,
    ModelHardwareRequirements,
    analyze_model_requirements,
    evaluate_compatibility,
)


def _req(**overrides):
    base = dict(
        model_name="Test MoE",
        total_model_size_gb=18.0,
        core_heads_size_gb=1.0,
        moe_experts_size_gb=17.0,
        ngram_size_gb=0.0,
        min_ram_required_gb=2.5,
        recommended_ram_gb=MIN_RAM_GB,
        recommended_vram_gb=MIN_VRAM_GB,
    )
    base.update(overrides)
    return ModelHardwareRequirements(**base)


def _profile(**overrides):
    base = dict(
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        used_ram_gb=4.0,
        ram_utilization_pct=25.0,
        gpu_detected=True,
        gpu_name="Test GPU",
        gpu_backend="test",
        total_vram_gb=4.0,
        free_vram_gb=4.0,
        has_discrete_gpu=True,
        cpu_count=8,
        platform_name="Linux",
    )
    base.update(overrides)
    return HardwareProfile(**base)


def _tiny_loader():
    # ~1 GB of core tensors, plenty of experts: recommended must hit the floor.
    return SimpleNamespace(
        tensors={
            "token_embd.weight": {"n_bytes": 512 * 1024 ** 2},
            "blk.0.attn_q.weight": {"n_bytes": 512 * 1024 ** 2},
            "blk.0.ffn_gate_exps.weight": {"n_bytes": 8 * 1024 ** 3},
        },
        metadata={"general.name": "Unit Test MoE"},
    )


def test_recommendations_are_floored_by_minimum_envelope():
    req = analyze_model_requirements(_tiny_loader())
    assert req.recommended_ram_gb == MIN_RAM_GB
    assert req.recommended_vram_gb == MIN_VRAM_GB


def test_budgets_are_adaptive_and_not_hardcoded():
    # A big machine gets more than the minimum floor.
    big = _profile(available_ram_gb=24.0, total_ram_gb=32.0, free_vram_gb=12.0, total_vram_gb=16.0)
    report = evaluate_compatibility(big, _req())
    assert report.effective_ram_budget_gb > MIN_RAM_GB
    assert report.effective_vram_budget_gb > MIN_VRAM_GB


def test_auto_budget_never_exceeds_available_ram():
    # Only ~2 GB free: must not overcommit the minimum 8 GB floor.
    tight = _profile(available_ram_gb=2.0, total_ram_gb=16.0)
    report = evaluate_compatibility(tight, _req())
    assert report.effective_ram_budget_gb <= 2.0


def test_below_envelope_is_flagged_and_constrained():
    small = _profile(total_ram_gb=4.0, available_ram_gb=3.0, total_vram_gb=1.0, free_vram_gb=1.0)
    report = evaluate_compatibility(small, _req())
    assert report.is_compatible is True
    assert report.status == "CONSTRAINED"
    assert any("minimum" in w for w in report.warnings)


def test_at_or_above_envelope_is_not_flagged():
    ok = _profile()  # 16 GB RAM / 4 GB VRAM
    report = evaluate_compatibility(ok, _req())
    assert not any("below the runtime's minimum" in w for w in report.warnings)
