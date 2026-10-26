"""Tests for RAM budget accounting.

Regression guard for the OOM: ``pipeline`` must size the expert cache from what
is left of the RAM budget *after* the resident core, never from the whole
budget (which oversubscribed host RAM).
"""
import torch
from engine.weights import (
    estimate_resident_bytes,
    is_expert_tensor,
    plan_ram_budgets,
)


class _Loader:
    def __init__(self, tensors):
        self.tensors = tensors


def test_is_expert_tensor():
    assert is_expert_tensor("blk.0.ffn_gate_exps.weight")
    assert is_expert_tensor("blk.7.ffn_down_exps.weight")
    assert not is_expert_tensor("blk.0.attn_q.weight")
    assert not is_expert_tensor("blk.0.ffn_gate_inp.weight")  # router is resident
    assert not is_expert_tensor("token_embd.weight")


def test_estimate_resident_counts_fp32_and_skips_experts():
    loader = _Loader({
        "blk.0.attn_q.weight": {"shape": [4, 8]},          # 32 elems -> 128 B
        "token_embd.weight": {"shape": [10, 2]},            # 20 -> 80 B
        "blk.0.ffn_gate_exps.weight": {"shape": [8, 4, 128]},  # skipped
    })
    # Default dtype is bfloat16 (2 bytes per element)
    assert estimate_resident_bytes(loader) == (32 + 20) * 2
    # Explicit FP32
    assert estimate_resident_bytes(loader, torch.float32) == (32 + 20) * 4


def test_plan_ram_budgets_subtracts_resident():
    loader = _Loader({"blk.0.attn_q.weight": {"shape": [100]}})  # 200 B resident (BF16)
    resident, expert = plan_ram_budgets(loader, 1000)
    assert resident == 200
    assert expert == 800  # not 1000


def test_plan_ram_budgets_never_negative():
    loader = _Loader({"blk.0.attn_q.weight": {"shape": [100]}})  # 200 B resident (BF16)
    resident, expert = plan_ram_budgets(loader, 100)
    assert resident == 200
    assert expert == 0
