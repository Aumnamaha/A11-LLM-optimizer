#!/usr/bin/env python3
"""Phase 1 model analyzer — static GGUF inventory for the Paged-MoE runtime.

Inspects a GGUF model *without running it* and answers the questions the pager
needs before any code is written:

    How many bytes belong to the MoE experts vs the resident core?
    How large is each expert / each tensor?
    How many logical pages does the model map to, and where do they live?

Outputs (default ``reports/``):

    model_inventory.json   full per-tensor inventory + aggregates
    page_table.jsonl       draft logical page table (metadata only)
    model_inventory.md     human-readable summary

Usage:
    python tools/model_analyzer/analyze_model.py --model models/Qwen3-30B-A3B
    python tools/model_analyzer/analyze_model.py --model models/Qwen3-30B-A3B --page-size-mib 1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig  # noqa: E402
from engine.loader import SplitGGUFLoader  # noqa: E402

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
ROUTER_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def parse_expert_name(name: str) -> Optional[Tuple[int, str]]:
    """Return ``(layer, kind)`` for an MoE expert tensor name, else ``None``."""
    match = EXPERT_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def parse_layer(name: str) -> Optional[int]:
    match = LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def classify(name: str) -> str:
    """Classify a tensor as ``expert`` (pageable) or ``resident``."""
    if parse_expert_name(name) is not None:
        return "expert"
    return "resident"


def _vocab_size(loader: SplitGGUFLoader) -> int:
    entry = loader.tensors.get("token_embd.weight")
    if entry and len(entry["shape"]) >= 2:
        return int(entry["shape"][1])
    return int(loader.metadata.get("tokenizer.ggml.vocab_size", 0)) or 0


def build_page_table(
    layer: int,
    kind: str,
    tensor_offset: int,
    n_bytes: int,
    n_experts: int,
    page_size: int,
    shard: str = "",
) -> List[Dict[str, Any]]:
    """Draft logical page table for one expert tensor (pure function).

    Each expert occupies a contiguous byte range; it is split into
    ``ceil(per_expert / page_size)`` fixed-size, independently addressable and
    independently decompressible pages. IDs are stable and human-debuggable:
    ``E073:L12:TUP:P000``.
    """
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    per_expert = n_bytes // n_experts
    pages_per_expert = max(1, math.ceil(per_expert / page_size))
    tag = kind.upper()
    entries: List[Dict[str, Any]] = []
    for expert in range(n_experts):
        base = tensor_offset + expert * per_expert
        for page in range(pages_per_expert):
            start = base + page * page_size
            length = min(page_size, per_expert - page * page_size)
            entries.append({
                "id": f"E{expert:03d}:L{layer:02d}:T{tag}:P{page:03d}",
                "layer": layer,
                "expert": expert,
                "tensor": kind,
                "page": page,
                "pages_per_expert": pages_per_expert,
                "byte_offset": start,
                "byte_length": length,
                "shard": shard,
            })
    return entries


def analyze(model_dir: str, page_size: int = 2 * 1024 * 1024) -> Dict[str, Any]:
    loader = SplitGGUFLoader(model_dir)
    vocab = _vocab_size(loader)
    try:
        cfg = ModelConfig.from_metadata(loader.metadata, vocab_size=vocab)
        n_experts = int(cfg.n_expert) or 128
        n_layer = int(cfg.n_layer)
    except Exception:  # metadata may be incomplete in synthetic tests
        cfg = None
        n_experts = int(loader.metadata.get("expert_count", 128))
        n_layer = 0

    resident: List[Dict[str, Any]] = []
    expert_tensors: List[Dict[str, Any]] = []
    per_layer: Dict[int, Dict[str, int]] = {}
    resident_bytes = 0
    expert_bytes = 0

    for name, meta in loader.tensors.items():
        layer = parse_layer(name)
        bucket_layer = per_layer.setdefault(layer if layer is not None else -1, {"resident": 0, "expert": 0})
        if classify(name) == "expert":
            lay, kind = parse_expert_name(name)
            record = {
                "name": name,
                "layer": lay,
                "kind": kind,
                "shard": meta["shard"],
                "shape": meta["shape"],
                "type": meta["type"],
                "n_bytes": meta["n_bytes"],
                "offset": meta["offset"],
                "per_expert_bytes": meta["n_bytes"] // n_experts if n_experts else 0,
            }
            expert_tensors.append(record)
            expert_bytes += meta["n_bytes"]
            bucket_layer["expert"] += meta["n_bytes"]
        else:
            record = {
                "name": name,
                "layer": layer,
                "shard": meta["shard"],
                "shape": meta["shape"],
                "type": meta["type"],
                "n_bytes": meta["n_bytes"],
                "offset": meta["offset"],
            }
            resident.append(record)
            resident_bytes += meta["n_bytes"]
            bucket_layer["resident"] += meta["n_bytes"]

    # Draft page table.
    page_entries: List[Dict[str, Any]] = []
    for tensor in sorted(expert_tensors, key=lambda t: (t["layer"], t["kind"])):
        page_entries.extend(build_page_table(
            tensor["layer"], tensor["kind"], tensor["offset"],
            tensor["n_bytes"], n_experts, page_size, tensor["shard"],
        ))

    total_bytes = resident_bytes + expert_bytes
    payload = sum(e["byte_length"] for e in page_entries)
    per_kind: Dict[str, int] = {}
    for tensor in expert_tensors:
        per_kind[tensor["kind"]] = per_kind.get(tensor["kind"], 0) + tensor["n_bytes"]

    report = {
        "model_dir": str(Path(model_dir).resolve()),
        "shards": [Path(s).name for s in loader.shards],
        "config": cfg.__dict__ if cfg is not None else {
            "n_layer": n_layer, "n_expert": n_experts,
        },
        "totals": {
            "total_bytes": total_bytes,
            "resident_bytes": resident_bytes,
            "expert_bytes": expert_bytes,
            "expert_percent": round(100.0 * expert_bytes / total_bytes, 2) if total_bytes else 0.0,
            "resident_percent": round(100.0 * resident_bytes / total_bytes, 2) if total_bytes else 0.0,
            "tensor_count": len(loader.tensors),
            "resident_tensor_count": len(resident),
            "expert_tensor_count": len(expert_tensors),
        },
        "expert_bytes_per_kind": per_kind,
        # Bytes for one expert of this kind *within a single layer* (the unit the
        # pager actually moves); guarded for synthetic inputs with no layer count.
        "per_layer_per_expert_bytes": {
            k: (v // (n_layer * n_experts)) if (n_layer and n_experts)
            else (v // n_experts if n_experts else 0)
            for k, v in per_kind.items()
        },
        "layers": [
            {"layer": lay, **bytes_}
            for lay, bytes_ in sorted(per_layer.items())
        ],
        "page_model": {
            "page_size_bytes": page_size,
            "page_size_mib": page_size / (1024 ** 2),
            "total_pages": len(page_entries),
            "pages_per_expert": page_entries[0]["pages_per_expert"] if page_entries else 0,
            "mapped_bytes": payload,
            "page_capacity_bytes": len(page_entries) * page_size,
            "page_table_bytes_estimate": len(page_entries) * 64,
            # Fraction of allocated page capacity that holds real payload. Low
            # values mean the page size wastes VRAM/RAM; high values mean pages
            # are well matched to the expert tensor sizes.
            "page_capacity_utilization": round(payload / (len(page_entries) * page_size), 4) if page_entries else 0.0,
        },
        "resident": sorted(resident, key=lambda r: -(r["n_bytes"] or 0)),
        "expert_tensors": sorted(expert_tensors, key=lambda r: (r["layer"], r["kind"])),
    }
    report["_page_entries"] = page_entries
    return report


def write_reports(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_entries = report.pop("_page_entries")

    (out_dir / "model_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "page_table.jsonl").open("w") as fh:
        for entry in page_entries:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    cfg = report["config"]
    totals = report["totals"]
    pm = report["page_model"]
    lines = [
        f"# Model inventory — `{Path(report['model_dir']).name}`",
        "",
        f"- Shards: {', '.join(report['shards'])}",
        f"- Layers: {cfg.get('n_layer')} · Experts: {cfg.get('n_expert')} · "
        f"Active/token: {cfg.get('n_expert_used')}",
        "",
        "## Size split",
        "",
        "| category | bytes | GiB | share |",
        "|---|---|---|---|",
        f"| resident core | {totals['resident_bytes']} | {totals['resident_bytes']/1024**3:.2f} | {totals['resident_percent']}% |",
        f"| MoE experts | {totals['expert_bytes']} | {totals['expert_bytes']/1024**3:.2f} | {totals['expert_percent']}% |",
        f"| **total** | {totals['total_bytes']} | {totals['total_bytes']/1024**3:.2f} | 100% |",
        "",
        "## Page model",
        "",
        f"- Page size: {pm['page_size_mib']:.0f} MiB",
        f"- Total logical pages: **{pm['total_pages']}** ({pm['pages_per_expert']} per expert tensor)",
        f"- Page-table estimate: {pm['page_table_bytes_estimate']} bytes (~{pm['page_table_bytes_estimate']/1024:.0f} KiB)",
        f"- Page capacity utilization: {pm['page_capacity_utilization']*100:.1f}%",
        "",
        "## Expert bytes per kind",
        "",
        "| kind | total (all layers) | per layer, per expert |",
        "|---|---|---|",
    ]
    for kind, total in report["expert_bytes_per_kind"].items():
        per = report["per_layer_per_expert_bytes"].get(kind, 0)
        lines.append(f"| {kind} | {total/1024**2:.2f} MiB | {per/1024**2:.3f} MiB |")
    lines += [
        "",
        "## Largest resident tensors",
        "",
        "| tensor | type | shape | MiB |",
        "|---|---|---|---|",
    ]
    for rec in report["resident"][:12]:
        lines.append(
            f"| `{rec['name']}` | {rec['type']} | {rec['shape']} | {rec['n_bytes']/1024**2:.1f} |"
        )
    lines.append("")
    (out_dir / "model_inventory.md").write_text("\n".join(lines))
    print(f"[analyzer] wrote {out_dir}/model_inventory.json, page_table.jsonl, model_inventory.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="GGUF file or directory")
    parser.add_argument("--page-size-mib", type=float, default=2.0, help="logical page size in MiB")
    parser.add_argument("--out", default="reports", help="output directory")
    args = parser.parse_args()

    model = Path(args.model).expanduser().resolve()
    if model.is_file():
        model = model.parent
    if not model.is_dir():
        raise SystemExit(f"model path not found: {model}")

    report = analyze(str(model), page_size=int(args.page_size_mib * 1024 ** 2))
    write_reports(report, Path(args.out).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
