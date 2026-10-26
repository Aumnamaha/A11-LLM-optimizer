#!/usr/bin/env python3
"""
Tree Speculative Sampling Throughput Benchmark

Measures:
  - Autoregressive Tokens/Second (Baseline)
  - Speculative Tree Tokens/Second (K=8 draft candidates)
  - Speedup Ratio S = TPS_spec / TPS_base
  - Peak System RAM & VRAM Usage via pynvml and psutil

Usage:
    python scripts/benchmark_throughput.py [--steps N] [--warmup M]

Environment: C:\\Users\\Aum Namaha\\Documents\\finalenv (torch, pynvml, psutil required)
"""

import argparse
import ctypes
import math
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Platform-specific: call into the C++ test binary to exercise the verifier
# ---------------------------------------------------------------------------

def get_test_binary_path() -> str:
    """Locate a11_tests.exe in the build directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "build", "bin", "a11_tests.exe"),
        os.path.join(project_root, "build", "Release", "a11_tests.exe"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def run_test_binary(test_filter: str = "") -> Tuple[str, int]:
    """Run the C++ test binary and capture stderr/stdout.

    Returns (combined_output, exit_code).
    """
    exe = get_test_binary_path()
    if not exe or not os.path.isfile(exe):
        return "[WARN] a11_tests.exe not found; skipping C++ benchmark", -1

    cmd = [exe]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout + result.stderr, result.returncode


# ---------------------------------------------------------------------------
# Hardware telemetry (pynvml / psutil)
# ---------------------------------------------------------------------------

def get_vram_usage_gb() -> float:
    """Return current VRAM used in GB via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 3)
    except Exception:
        return -1.0


def get_system_ram_gb() -> float:
    """Return current system RAM used in GB via psutil."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024 ** 3)
    except Exception:
        return -1.0


def get_gpu_info_string() -> str:
    """Return GPU name string."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        return str(name) if isinstance(name, str) else name.decode("utf-8")
    except Exception:
        return "Unknown GPU"


# ---------------------------------------------------------------------------
# Speculative verification throughput benchmark (C++ binding via test binary)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    strategy: str = ""
    tokens_per_second: float = 0.0
    latency_ms_per_token: float = 0.0
    peak_vram_gb: float = -1.0
    peak_system_ram_gb: float = -1.0
    total_tokens_generated: int = 0
    notes: str = ""


def benchmark_autoregressive(
    steps: int = 200, vocab_size: int = 32000
) -> BenchmarkResult:
    """Baseline: simulate one-token-at-a-time generation.

    Each step: sample one token from a simulated distribution (no real model).
    Measures wall-clock time for the full sequence.
    """
    peak_vram = get_vram_usage_gb()
    peak_ram  = get_system_ram_gb()

    # Simulate autoregressive loop with synthetic work proportional to context length
    seq_len = 128  # tokens in context

    start = time.perf_counter()
    for step in range(steps):
        # Synthetic computation: matrix-vector multiply of size (seq_len x hidden)
        hidden = 4096
        n_ops = seq_len * hidden * 2  # FLOPs per token generation step
        # Simulate the compute by doing actual work on a small tensor
        import torch
        if torch.cuda.is_available():
            x = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
            _ = torch.nn.functional.silu(x @ torch.randn(hidden, hidden, device="cuda", dtype=torch.float32))
        seq_len += 1  # Context grows

    elapsed = time.perf_counter() - start
    tokens_generated = steps

    tps = tokens_generated / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / tokens_generated) * 1000.0 if tokens_generated > 0 else 0.0

    peak_vram = max(peak_vram, get_vram_usage_gb())
    peak_ram  = max(peak_ram, get_system_ram_gb())

    return BenchmarkResult(
        strategy="Autoregressive (Baseline)",
        tokens_per_second=tps,
        latency_ms_per_token=latency_ms,
        peak_vram_gb=round(peak_vram, 2),
        peak_system_ram_gb=round(peak_ram, 2),
        total_tokens_generated=tokens_generated,
    )


def benchmark_speculative(
    steps: int = 50, k: int = 8, vocab_size: int = 32000
) -> BenchmarkResult:
    """Speculative tree verification with K=8 draft candidates.

    Each step: generate K draft tokens, verify against target in one pass,
    accept/reject based on probability matching.
    """
    peak_vram = get_vram_usage_gb()
    peak_ram  = get_system_ram_gb()

    import torch

    hidden = 4096
    seq_len = 128

    # Pre-allocate device buffers for the verification kernel
    if torch.cuda.is_available():
        d_hidden = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        d_draft_logits = torch.zeros(k, vocab_size // 64, device="cuda", dtype=torch.float32)
    else:
        d_hidden = None
        d_draft_logits = None

    total_accepted_tokens = 0
    start = time.perf_counter()

    for step in range(steps):
        # Generate K draft tokens (simulated from a smaller model)
        draft_probs = torch.softmax(torch.randn(k, vocab_size // 64, device="cuda" if d_hidden is not None else "cpu", dtype=torch.float32), dim=-1)

        # Target verification: one forward pass for all K candidates
        if d_hidden is not None:
            W = torch.randn(vocab_size // 64, hidden, device="cuda", dtype=torch.float32)
            target_logits = d_hidden @ W.T  # shape (1, vocab//64)
        else:
            target_logits = torch.randn(k, vocab_size // 64, dtype=torch.float32)

        # Rejection sampling acceptance — generate per-candidate scores
        draft_log_probs = torch.log(draft_probs + 1e-10).mean(dim=-1)  # shape (k,) if batched
        target_log_probs = torch.softmax(target_logits, dim=-1).mean(dim=-1)
        # Expand to k candidates by repeating the single score
        if target_log_probs.numel() == 1:
            target_log_probs = target_log_probs.expand(k)

        accepted = 0
        for i in range(k):
            accept_prob = min(1.0, float(target_log_probs[i]) / max(float(draft_probs[i].mean()), 1e-10))
            if torch.rand(1).item() < accept_prob:
                accepted += 1
            else:
                break

        total_accepted_tokens += accepted
        seq_len += accepted + 1  # Accepted drafts + one new target token

    elapsed = time.perf_counter() - start

    effective_tps = total_accepted_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / steps) * 1000.0 if steps > 0 else 0.0

    peak_vram = max(peak_vram, get_vram_usage_gb())
    peak_ram  = max(peak_ram, get_system_ram_gb())

    return BenchmarkResult(
        strategy=f"Speculative Tree (K={k})",
        tokens_per_second=effective_tps,
        latency_ms_per_token=latency_ms,
        peak_vram_gb=round(peak_vram, 2),
        peak_system_ram_gb=round(peak_ram, 2),
        total_tokens_generated=total_accepted_tokens,
    )


# ---------------------------------------------------------------------------
# C++ SpeculativeVerifier diagnostic benchmark (via test binary)
# ---------------------------------------------------------------------------

def benchmark_cpp_verifier(steps: int = 50) -> BenchmarkResult:
    """Exercise the C++ SpeculativeTreeVerifier via the test binary.

    Measures wall-clock time for running all tests, then parses diagnostic
    counters from stderr output to compute tokens/second.
    """
    peak_vram = get_vram_usage_gb()
    peak_ram  = get_system_ram_gb()

    # Run the test binary and capture output for diagnostic parsing
    t_start = time.perf_counter()
    output, rc = run_test_binary()
    elapsed = time.perf_counter() - t_start

    if rc != 0:
        return BenchmarkResult(
            strategy="C++ SpeculativeVerifier (binary unavailable)",
            notes=f"Test binary returned code {rc}: {output[:200]}",
            peak_vram_gb=round(peak_vram, 2),
            peak_system_ram_gb=round(peak_ram, 2),
        )

    # Parse diagnostic counters from output (stderr contains [SpeculativeTree] lines)
    total_accepted = 0
    fallback_count = 0
    for line in output.split("\n"):
        if "Verification: accepted=" in line:
            try:
                val = int(line.split("accepted=")[1].split(",")[0])
                total_accepted += val
            except (ValueError, IndexError):
                pass
        if "Fallback to autoregressive" in line:
            fallback_count += 1

    tps = total_accepted / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / steps * 1000.0) if steps > 0 else 0.0

    peak_vram = max(peak_vram, get_vram_usage_gb())
    peak_ram  = max(peak_ram, get_system_ram_gb())

    return BenchmarkResult(
        strategy="C++ SpeculativeVerifier (native)",
        tokens_per_second=tps,
        latency_ms_per_token=latency_ms,
        peak_vram_gb=round(peak_vram, 2),
        peak_system_ram_gb=round(peak_ram, 2),
        total_tokens_generated=total_accepted,
        notes=f"Fallbacks: {fallback_count}, Elapsed: {elapsed:.3f}s",
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_table(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as an aligned Markdown table."""
    header = "| Strategy | Tokens/sec | Latency (ms/tok) | Peak VRAM (GB) | Peak RAM (GB) | Total Tokens |"
    sep = "|---|---|---|---|---|---|"

    lines = [header, sep]
    for r in results:
        tps_str = f"{r.tokens_per_second:.1f}" if r.tokens_per_second > 0 else "N/A"
        lat_str = f"{r.latency_ms_per_token:.2f}" if r.latency_ms_per_token > 0 else "N/A"
        vram_str = f"{r.peak_vram_gb:.2f}" if r.peak_vram_gb >= 0 else "N/A"
        ram_str = f"{r.peak_system_ram_gb:.2f}" if r.peak_system_ram_gb >= 0 else "N/A"
        notes = f" | {r.notes}" if r.notes and len(r.notes) > 0 else ""
        lines.append(
            f"| {r.strategy} | {tps_str} | {lat_str} | {vram_str} | {ram_str} "
            f"| {r.total_tokens_generated}{notes} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Tree Speculative Sampling Throughput Benchmark")
    parser.add_argument("--steps", type=int, default=200, help="Number of generation steps (autoregressive)")
    parser.add_argument("--spec-steps", type=int, default=50, help="Number of speculation steps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing")
    args = parser.parse_args()

    print("=" * 72)
    print("A11 Speculative Sampling Throughput Benchmark")
    print("=" * 72)

    # GPU info
    gpu_name = get_gpu_info_string()
    cuda_avail = False
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except Exception:
        pass

    print(f"GPU: {gpu_name}")
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        import torch
        print(f"  CUDA Device 0: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        get_vram_usage_gb()
        get_system_ram_gb()
    print()

    results: List[BenchmarkResult] = []

    # 1. Autoregressive baseline
    print("[1/3] Running autoregressive baseline...")
    base_result = benchmark_autoregressive(steps=args.steps)
    results.append(base_result)
    print(f"      TPS: {base_result.tokens_per_second:.2f} | Latency: {base_result.latency_ms_per_token:.2f} ms/tok")
    print()

    # 2. Speculative tree verification (Python simulation)
    print("[2/3] Running speculative tree benchmark (K=8)...")
    spec_result = benchmark_speculative(steps=args.spec_steps, k=8)
    results.append(spec_result)
    speedup = (spec_result.tokens_per_second / base_result.tokens_per_second
               if base_result.tokens_per_second > 0 else 0.0)
    print(f"      TPS: {spec_result.tokens_per_second:.2f} | Speedup: {speedup:.2f}x")
    print()

    # 3. C++ native verifier benchmark
    print("[3/3] Running C++ SpeculativeVerifier benchmark...")
    cpp_result = benchmark_cpp_verifier(steps=args.spec_steps)
    results.append(cpp_result)
    if cpp_result.tokens_per_second > 0:
        cpp_speedup = (cpp_result.tokens_per_second / base_result.tokens_per_second
                       if base_result.tokens_per_second > 0 else 0.0)
        print(f"      TPS: {cpp_result.tokens_per_second:.2f} | Speedup vs baseline: {cpp_speedup:.2f}x")
    print()

    # Summary table
    print("=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    print(format_table(results))
    print()

    if base_result.tokens_per_second > 0 and spec_result.tokens_per_second > 0:
        print(f"Speedup Ratio (Speculative / Autoregressive): {speedup:.2f}x")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
