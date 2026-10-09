#!/usr/bin/env python3
"""Weight-parity checker for :class:`engine.weights.WeightStore`.

Compares tensors served by ``WeightStore`` against an independent reference:
``gguf.quants.dequantize`` applied to the full GGUF tensor.

GGUF/ggml stores tensors with ``ne0`` (first dim) contiguous, so a ``shape`` of
``[in, out, n_experts]`` maps to a flat buffer laid out in Fortran order. The
reference therefore reshapes with ``order="F"`` as well.

Usage:
    python scripts/verify_weights_parity.py [MODEL_DIR]

Exit code is non-zero if any tensor mismatches.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from engine.loader import SplitGGUFLoader  # noqa: E402
from engine.mmap_loader import DirectSSDMmapManager  # noqa: E402
from engine.weights import WeightStore  # noqa: E402

ATOL = 1e-3
_results: list[bool] = []


def ref_tensor(tensor) -> np.ndarray:
    """Independent dequantization: gguf-py's canonical [out, in] layout.

    ``dequantize`` returns reversed(dims) in C order (the PyTorch-native weight
    layout), so no extra reshape is needed. F32 ``tensor.data`` is already in
    that layout.
    """
    if tensor.tensor_type.name == "F32":
        return tensor.data
    return dequantize(tensor.data, tensor.tensor_type)


def check(label: str, ours, ref) -> None:
    ref = np.asarray(ref, dtype=np.float32).reshape(ours.shape)
    diff = float(np.abs(ours.numpy() - ref).max())
    ok = np.isfinite(diff) and diff <= ATOL
    _results.append(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {label:<34s} maxdiff={diff:.2e} shape={tuple(ours.shape)}")


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "models", "Qwen3-30B-A3B"
    )
    shards = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    if not shards:
        print(f"No .gguf files in {model_dir}", file=sys.stderr)
        return 2

    reader = gguf.GGUFReader(shards[0])
    by_name = {t.name: t for t in reader.tensors}

    loader = SplitGGUFLoader(model_dir)
    mmap_mgr = DirectSSDMmapManager(loader)
    n_experts = int(loader.metadata.get(f"{loader.metadata.get('general.architecture')}.expert_count", 128))
    store = WeightStore(loader, mmap_mgr, n_experts=n_experts)

    print("\n=== dense tensor parity ===")
    for name in (
        "blk.0.attn_norm.weight",   # F32
        "blk.0.attn_q.weight",      # Q4_K
        "blk.0.attn_v.weight",      # Q6_K
        "blk.0.ffn_gate_inp.weight",  # router
    ):
        if name in by_name:
            check(name, store.get_torch(name), ref_tensor(by_name[name]))

    print("\n=== MoE expert parity ===")
    for layer, experts in ((0, (0, 64, 127)), (24, (1, 120))):
        for kind in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            if name not in by_name:
                continue
            ref_full = ref_tensor(by_name[name])  # [n_experts, out, in]
            for e in experts:
                if e >= n_experts:
                    continue
                check(f"blk.{layer}.{kind}_exps[{e}]",
                      store.get_expert(layer, kind, e), ref_full[e])
            del ref_full

    print(f"\nexpert cache after run: entries={store.expert_cache_stats[0]} "
          f"bytes={store.expert_cache_stats[1]}")
    mmap_mgr.close()

    failed = _results.count(False)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
