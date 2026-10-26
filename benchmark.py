#!/usr/bin/env python3
"""Reproducible end-to-end benchmark for the A11 GGUF inference pipeline.

TTFT includes prompt processing and ends when the first output token is yielded.
Generation throughput is generated tokens divided by elapsed time from the start
of inference through the last yielded token. Initialization is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch

from engine.pipeline import A11Pipeline


DEFAULT_PROMPT = (
    "Write a Python script that parses a nested JSON API response and stores "
    "the results in SQLite."
)


def model_directory(model_path: str) -> str:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Model file must be GGUF: {path}")
        return str(path.parent)
    if not path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    return str(path)


def run_benchmark(
    model_path: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 150,
    vram_gb: float = 4.0,
    ram_gb: float = 12.0,
    seed: int = 42,
) -> dict:
    if max_tokens <= 0:
        raise ValueError("--tokens must be greater than zero")
    random.seed(seed)
    torch.manual_seed(seed)

    resolved_model = model_directory(model_path)
    print("\n[Benchmark] Initializing A11 pipeline...", flush=True)
    init_start = time.perf_counter()
    pipeline = A11Pipeline(
        resolved_model,
        vram_budget_gb=vram_gb,
        ram_budget_gb=ram_gb,
    )
    init_seconds = time.perf_counter() - init_start
    print(f"[Benchmark] Initialization: {init_seconds:.3f}s")
    print(f"[Benchmark] Prompt: {prompt}\n[Benchmark] Generating up to {max_tokens} tokens...\n")

    token_count = 0
    ttft_seconds = None
    inference_start = time.perf_counter()
    last_token_time = inference_start
    interrupted = False
    try:
        for token_text in pipeline.stream_inference(prompt, max_new_tokens=max_tokens):
            now = time.perf_counter()
            if ttft_seconds is None:
                ttft_seconds = now - inference_start
            token_count += 1
            last_token_time = now
            sys.stdout.write(token_text)
            sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Benchmark] Interrupted.")
    finally:
        stats = pipeline.get_cache_stats()
        pipeline.shutdown()

    elapsed_seconds = max(0.0, last_token_time - inference_start)
    throughput = token_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    result = {
        "model_path": resolved_model,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_characters": len(prompt),
        "max_tokens": max_tokens,
        "seed": seed,
        "initialization_seconds": round(init_seconds, 6),
        "ttft_seconds": round(ttft_seconds, 6) if ttft_seconds is not None else None,
        "generation_elapsed_seconds": round(elapsed_seconds, 6),
        "tokens_generated": token_count,
        "tokens_per_second": round(throughput, 4),
        "interrupted": interrupted,
        "cache": stats,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
    }
    print("\n\n" + "=" * 56)
    print("A11 BENCHMARK RESULTS")
    print("=" * 56)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model directory or a GGUF shard file")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fixed prompt for repeatable runs")
    parser.add_argument("--tokens", type=int, default=150, help="Maximum output tokens")
    parser.add_argument("--vram", type=float, default=None, help="Requested VRAM budget in GB (default: auto-detect)")
    parser.add_argument("--ram", type=float, default=None, help="Requested RAM cache budget in GB (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--json-out", help="Optional path to write the JSON result")
    args = parser.parse_args()

    try:
        result = run_benchmark(
            args.model, args.prompt, args.tokens, args.vram, args.ram, args.seed
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[Benchmark] JSON report saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
