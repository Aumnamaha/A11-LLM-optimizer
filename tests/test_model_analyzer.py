"""Hermetic tests for the Phase 1 model analyzer (no model files required)."""
import pytest

from tools.model_analyzer.analyze_model import (
    build_page_table,
    classify,
    parse_expert_name,
    parse_layer,
)


def test_parse_expert_name_matches_moe_tensors():
    assert parse_expert_name("blk.12.ffn_up_exps.weight") == (12, "up")
    assert parse_expert_name("blk.0.ffn_gate_exps.weight") == (0, "gate")
    assert parse_expert_name("blk.47.ffn_down_exps.weight") == (47, "down")


def test_parse_expert_name_ignores_non_experts():
    assert parse_expert_name("blk.0.attn_q.weight") is None
    assert parse_expert_name("blk.0.ffn_gate_inp.weight") is None
    assert parse_expert_name("token_embd.weight") is None


def test_classify_separates_resident_from_pageable():
    assert classify("blk.3.ffn_gate_exps.weight") == "expert"
    assert classify("blk.3.ffn_up_exps.weight") == "expert"
    assert classify("blk.3.ffn_down_exps.weight") == "expert"
    assert classify("blk.3.attn_norm.weight") == "resident"
    assert classify("blk.3.ffn_gate_inp.weight") == "resident"
    assert classify("output.weight") == "resident"


def test_parse_layer():
    assert parse_layer("blk.24.attn_q.weight") == 24
    assert parse_layer("token_embd.weight") is None


def test_build_page_table_splits_experts_into_stable_pages():
    # 4 experts x 100 bytes = 400 bytes, 64-byte pages -> 2 pages per expert.
    entries = build_page_table(
        layer=0, kind="gate", tensor_offset=1000,
        n_bytes=400, n_experts=4, page_size=64,
    )
    assert len(entries) == 8
    assert entries[0] == {
        "id": "E000:L00:TGATE:P000",
        "layer": 0, "expert": 0, "tensor": "gate", "page": 0,
        "pages_per_expert": 2, "byte_offset": 1000, "byte_length": 64, "shard": "",
    }
    # Second page of expert 0 is the partial tail.
    assert entries[1]["byte_offset"] == 1064
    assert entries[1]["byte_length"] == 36
    # Expert 1 starts a fresh contiguous range.
    assert entries[2]["byte_offset"] == 1100
    assert entries[2]["byte_length"] == 64
    assert entries[-1]["id"] == "E003:L00:TGATE:P001"
    assert entries[-1]["byte_offset"] == 1364


def test_build_page_table_requires_positive_experts():
    with pytest.raises(ValueError):
        build_page_table(0, "up", 0, 128, 0, 64)


def test_build_page_table_single_page_per_expert_when_small():
    entries = build_page_table(7, "down", 4096, n_bytes=512 * 4, n_experts=4, page_size=4096)
    assert len(entries) == 4
    assert all(e["pages_per_expert"] == 1 for e in entries)
    assert [e["byte_offset"] for e in entries] == [4096, 4608, 5120, 5632]
    assert all(e["byte_length"] == 512 for e in entries)
