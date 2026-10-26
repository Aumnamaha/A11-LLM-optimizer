#!/usr/bin/env python3
"""Phase 0 routing trace — measures MoE expert locality without modifying the engine.

The pager is only worth building if expert selection is *local* enough to cache.
This tool runs the (already real) :class:`engine.pipeline.A11Pipeline` over a few
prompts and records which experts the router selects, then reports locality:

    temporal Jaccard     overlap of a layer's experts between adjacent tokens
    consecutive-layer J   overlap of adjacent layers' experts within one token
    hot-set coverage     #experts covering 50% / 90% of all selections
    per-layer usage      unique experts and the busiest experts per layer

It wraps ``pipeline.router.select_experts`` at runtime (no engine edits) and only
does *prefill* (no sampling), so it stays as cheap as the model allows.

Outputs ``baseline/routing.json``.

Usage:
    python tools/model_analyzer/routing_trace.py --model models/Qwen3-30B-A3B --max-layers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROMPTS = [
    "Write a Python script that parses a nested JSON API response and stores the results in SQLite.",
    "Explain how a buck converter regulates output voltage.",
    "Write a Python script that parses a nested JSON API response and stores the results in SQLite.",
]


def compute_stats(trace: List[Dict[str, Any]], num_layers: int) -> Dict[str, Any]:
    """Pure locality statistics over a routing trace (testable in isolation)."""
    counts: Counter = Counter()
    per_layer: Dict[int, set] = defaultdict(set)
    per_token: Dict[Any, Dict[int, set]] = {}
    for rec in trace:
        experts = set(rec["experts"])
        per_layer[rec["layer"]] |= experts
        for exp in experts:
            counts[(rec["layer"], exp)] += 1
        per_token.setdefault((rec["prompt"], rec["token"]), {})[rec["layer"]] = experts

    total = sum(counts.values())
    ordered = sorted(counts.values(), reverse=True)

    def hot_set(fraction: float) -> int:
        acc = 0
        for i, value in enumerate(ordered, 1):
            acc += value
            if acc >= fraction * total:
                return i
        return len(ordered)

    inter = union = 0
    for layers in per_token.values():
        for layer in range(num_layers - 1):
            a, b = layers.get(layer), layers.get(layer + 1)
            if a is None or b is None:
                continue
            inter += len(a & b)
            union += len(a | b)
    layer_jaccard = inter / union if union else 0.0

    by_prompt: Dict[Any, Dict[int, Dict[int, set]]] = defaultdict(lambda: defaultdict(dict))
    for (prompt, token), layers in per_token.items():
        for layer, experts in layers.items():
            by_prompt[prompt][layer][token] = experts
    inter = union = 0
    for layers in by_prompt.values():
        for tokens in layers.values():
            ts = sorted(tokens)
            for a, b in zip(ts, ts[1:]):
                inter += len(tokens[a] & tokens[b])
                union += len(tokens[a] | tokens[b])
    temporal_jaccard = inter / union if union else 0.0

    return {
        "selections": total,
        "tokens_traced": len(per_token),
        "unique_layer_expert_pairs": len(counts),
        "unique_experts_all_layers": len({exp for (_, exp) in counts}),
        "hot_set_50pct": hot_set(0.50),
        "hot_set_90pct": hot_set(0.90),
        "consecutive_layer_jaccard": round(layer_jaccard, 4),
        "temporal_token_jaccard": round(temporal_jaccard, 4),
        "per_layer_unique_experts": {str(layer): len(experts) for layer, experts in sorted(per_layer.items())},
    }


def run_trace(model_dir: str, prompts: List[str], max_tokens: int,
              max_layers: int, vram_gb: Any, ram_gb: Any) -> Dict[str, Any]:
    from engine.pipeline import A11Pipeline

    pipeline = A11Pipeline(model_dir, vram_budget_gb=vram_gb, ram_budget_gb=ram_gb,
                           max_layers=max_layers)
    num_layers = pipeline.num_layers

    trace: List[Dict[str, Any]] = []
    state = {"call": 0, "prompt": 0}
    original = pipeline.router.select_experts

    def wrapped(hidden_states, router_weights):
        indices, weights = original(hidden_states, router_weights)
        call = state["call"]
        state["call"] += 1
        trace.append({
            "prompt": state["prompt"],
            "token": call // num_layers,
            "layer": call % num_layers,
            "experts": [int(x) for x in indices[0].tolist()],
        })
        return indices, weights

    pipeline.router.select_experts = wrapped  # type: ignore[assignment]

    per_prompt = []
    started = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        token_ids = pipeline.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [pipeline.tokenizer.encode("Hello")[0]]
        token_ids = token_ids[:max_tokens]
        state["call"] = 0
        state["prompt"] = prompt_index
        kv_cache: Dict[int, Any] = {}
        t0 = time.perf_counter()
        for pos, token_id in enumerate(token_ids):
            pipeline.forward_token(token_id, pos, kv_cache)
        per_prompt.append({
            "prompt_index": prompt_index,
            "prompt": prompt[:120],
            "tokens": len(token_ids),
            "seconds": round(time.perf_counter() - t0, 3),
        })
    elapsed = time.perf_counter() - started

    stats = compute_stats(trace, num_layers)
    report = {
        "model_dir": str(Path(model_dir).resolve()),
        "num_layers_traced": num_layers,
        "n_experts": int(pipeline.config.n_expert),
        "top_k": int(pipeline.config.n_expert_used),
        "elapsed_seconds": round(elapsed, 2),
        "per_prompt": per_prompt,
        "totals": stats,
        "records": trace,
    }
    try:
        pipeline.shutdown()
    except Exception:
        pass
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="baseline/routing.json")
    parser.add_argument("--tokens", type=int, default=8, help="prefill tokens per prompt")
    parser.add_argument("--max-layers", type=int, default=48, help="limit traced layers (speed)")
    parser.add_argument("--prompts", help="JSON file with a list of prompt strings")
    parser.add_argument("--vram", type=float, default=None, help="VRAM budget in GB (default: auto-detect)")
    parser.add_argument("--ram", type=float, default=None, help="RAM cache budget in GB (default: auto-detect)")
    parser.add_argument("--no-records", action="store_true", help="omit the raw per-layer records")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())

    report = run_trace(args.model, prompts, args.tokens, args.max_layers, args.vram, args.ram)
    if args.no_records:
        report.pop("records", None)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== routing locality ===")
    for key, value in report["totals"].items():
        if key != "per_layer_unique_experts":
            print(f"  {key}: {value}")
    print(f"[routing] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
