#!/usr/bin/env python3
"""Weight-parity checker for :class:`engine.weights.WeightStore`.

Compares tensors served by ``WeightStore`` against an independent reference:
``gguf.quants.dequantize`` applied to the full GGUF tensor.

GGUF/ggml stores tensors with ``ne0`` (first dim) contiguous. The checker reads
only each selected expert's quantized byte range and dequantizes that range.

Usage:
    python scripts/verify_weights_parity.py [MODEL_DIR]

Exit code is non-zero if any tensor mismatches.
"""
from __future__ import annotations

import glob
import mmap
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

import torch  # noqa: E402

from engine.loader import SplitGGUFLoader  # noqa: E402
from engine.mmap_loader import DirectSSDMmapManager  # noqa: E402
from engine.pager import MemoryPager, ModelPageTable  # noqa: E402
from engine.weights import WeightStore  # noqa: E402

ATOL = 2e-2  # BF16 has ~1e-2 precision (7 mantissa bits)
_results: list[bool] = []


def ref_tensor(tensor, mm: mmap.mmap) -> np.ndarray:
    """Independently dequantize one complete dense tensor."""
    raw = np.frombuffer(mm, dtype=np.uint8, count=tensor.n_bytes,
                        offset=tensor.data_offset)
    if tensor.tensor_type.name == "F32":
        flat = raw.view(np.float32)
    else:
        flat = dequantize(raw, tensor.tensor_type)
    return flat.reshape(tuple(reversed(tensor.shape.tolist())))

def ref_expert(tensor, mm: mmap.mmap, expert_idx: int, n_experts: int) -> np.ndarray:
    """Dequantize only one expert, avoiding a full 128-expert allocation."""
    per_expert = tensor.n_bytes // n_experts
    start = tensor.data_offset + expert_idx * per_expert
    raw = np.frombuffer(mm, dtype=np.uint8, count=per_expert, offset=start)
    flat = dequantize(raw, tensor.tensor_type)
    return flat.reshape(tuple(reversed(tensor.shape.tolist()[:2])))


def check(label: str, ours, ref) -> None:
    ref = np.asarray(ref, dtype=np.float32).reshape(ours.shape)
    # Convert BF16 tensors to float32 for numpy comparison
    if hasattr(ours, 'dtype') and ours.dtype == torch.bfloat16:
        ours_np = ours.to(torch.float32).numpy()
    else:
        ours_np = ours.numpy()
    diff = float(np.abs(ours_np - ref).max())
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

    readers = [gguf.GGUFReader(path) for path in shards]
    maps = {}
    for path in shards:
        f = open(path, "rb")
        maps[path] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
    by_name = {}
    tensor_maps = {}
    for path, reader in zip(shards, readers):
        for tensor in reader.tensors:
            by_name[tensor.name] = tensor
            tensor_maps[tensor.name] = maps[path][1]

    loader = SplitGGUFLoader(model_dir)
    mmap_mgr = DirectSSDMmapManager(loader)
    n_experts = int(loader.metadata.get(f"{loader.metadata.get('general.architecture')}.expert_count", 128))

    # Use BF16 for compute and resident core (matches pipeline)
    compute_dtype = torch.bfloat16

    # Create pager (same as pipeline)
    page_table = ModelPageTable("reports/page_table.jsonl", loader)
    # Use a reasonable slot budget (same logic as pipeline)
    ram_budget_bytes = 8 * 1024 ** 3
    resident_bf16 = 2.87 * 1024 ** 3
    expert_cache_bytes = max(0, ram_budget_bytes - resident_bf16)
    dequantized_expert_bytes = 3 * 1024 * 1024  # BF16 expert pages
    n_slots_ram = max(1, int(expert_cache_bytes // dequantized_expert_bytes))
    n_slots_vram = 0  # parity runs on CPU
    pager = MemoryPager(page_table, loader, mmap_mgr, n_slots_ram=n_slots_ram, n_slots_vram=n_slots_vram, compute_dtype=compute_dtype)

    store = WeightStore(loader, mmap_mgr, pager=pager, n_experts=n_experts, resident_dtype=compute_dtype)

    print("\n=== dense tensor parity ===")
    for name in (
        "blk.0.attn_norm.weight",   # F32
        "blk.0.attn_q.weight",      # Q4_K
        "blk.0.attn_v.weight",      # Q6_K
        "blk.0.ffn_gate_inp.weight",  # router
    ):
        if name in by_name:
            check(name, store.get_torch(name), ref_tensor(by_name[name], tensor_maps[name]))

    print("\n=== MoE expert parity ===")
    for layer, experts in ((0, (0, 64, 127)), (24, (1, 120))):
        for kind in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            if name not in by_name:
                continue
            for e in experts:
                if e >= n_experts:
                    continue
                ref = ref_expert(by_name[name], tensor_maps[name], e, n_experts)
                check(f"blk.{layer}.{kind}_exps[{e}]",
                      store.get_expert(layer, kind, e), ref)

    if hasattr(store, "pager") and store.pager is not None:
        pager_stats = store.pager.stats
        print(f"\npager stats: faults={int(pager_stats.get('ssd_page_faults',0))} "
              f"hits={int(pager_stats.get('ram_hits',0))} "
              f"evictions={int(pager_stats.get('evictions',0))} "
              f"decomp_ms={pager_stats.get('decompression_time_ms',0):.1f}")
    else:
        print(f"\nexpert cache after run: entries={store.expert_cache_stats[0]} "
              f"bytes={store.expert_cache_stats[1]}")
    mmap_mgr.close()
    for f, mm in maps.values():
        mm.close()
        f.close()

    failed = _results.count(False)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
