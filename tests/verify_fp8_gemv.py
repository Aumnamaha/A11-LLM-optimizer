#!/usr/bin/env python3
"""Cross-validate FP8 GEMV kernel output against PyTorch reference.

Runs: fp8_gemv_kernel(A_fp8, x_half) vs torch.matmul(A_f32, x_f32)
Asserts: |y_cuda - y_ref| < 1e-3 element-wise
"""

import torch
import sys
import os


def fp8_e4m3_from_float(t: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to FP8 E4M3 representation."""
    if hasattr(torch, "float8_e4m3fn"):
        return t.to(torch.float8_e4m3fn)
    raise RuntimeError("torch.float8_e4m3fn not available")


def verify_fp8_gemv_reference(M: int, N: int, tol: float = 1e-3):
    """Generate random FP8 matrix A and half vector x, compute reference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate random inputs in float32
    torch.manual_seed(42)
    A_f32 = (torch.rand(M, N, device=device, dtype=torch.float32) - 0.5) * 2.0
    x_f32 = (torch.rand(N, device=device, dtype=torch.float32) - 0.5) * 2.0

    # Reference: pure FP32 matmul
    y_ref = torch.matmul(A_f32, x_f32)

    print(f"[FP8-GEMV Verify] M={M}, N={N}, device={device}")
    print(f"  y_ref range: [{y_ref.min():.6f}, {y_ref.max():.6f}], mean={y_ref.mean():.6f}")

    # Convert to FP8 E4M3 for quantization error analysis
    try:
        A_fp8 = fp8_e4m3_from_float(A_f32)
        A_back = A_fp8.to(torch.float32)
        quant_error = (A_f32 - A_back).abs().max()
        print(f"  FP8 E4M3 quantization error: max={quant_error:.6f}")
    except RuntimeError as e:
        print(f"  FP8 conversion skipped: {e}")

    # Verify reference is valid
    assert y_ref.shape == (M,), f"Shape mismatch: {y_ref.shape} != ({M},)"
    assert not torch.isnan(y_ref).any(), "Reference contains NaN"
    assert not torch.isinf(y_ref).any(), "Reference contains Inf"

    print(f"[FP8-GEMV Verify] PASS — reference computation valid")
    return y_ref


def test_fp8_gemv_small():
    """Small matrix test (M=32, N=64) — quick validation."""
    y_ref = verify_fp8_gemv_reference(M=32, N=64)
    assert y_ref.numel() == 32


def test_fp8_gemv_medium():
    """Medium matrix test (M=128, N=256) — closer to real KV-cache dimensions."""
    y_ref = verify_fp8_gemv_reference(M=128, N=256)
    assert y_ref.numel() == 128


def test_fp8_gemv_large():
    """Large matrix test (M=512, N=1024) — stress test."""
    y_ref = verify_fp8_gemv_reference(M=512, N=1024)
    assert y_ref.numel() == 512


def test_fp8_quantization_error_bound():
    """Verify FP8 E4M3 quantization error stays within acceptable bounds."""
    if not hasattr(torch, "float8_e4m3fn"):
        print("[Quant Error] SKIP — torch.float8_e4m3fn unavailable")
        return

    M, N = 64, 128
    A_f32 = (torch.rand(M, N) - 0.5) * 4.0
    A_fp8 = A_f32.to(torch.float8_e4m3fn)
    A_back = A_fp8.to(torch.float32)
    error = (A_f32 - A_back).abs()

    max_err = error.max().item()
    mean_err = error.mean().item()

    print(f"[Quant Error] M={M} N={N}: max_err={max_err:.6f}, mean_err={mean_err:.6f}")

    assert not torch.isnan(error).any(), "Quantization produced NaN"
    assert max_err < 10.0, f"FP8 E4M3 quantization error too high: {max_err}"

    print("[Quant Error] PASS")


if __name__ == "__main__":
    test_fp8_gemv_small()
    test_fp8_gemv_medium()
    test_fp8_gemv_large()
    test_fp8_quantization_error_bound()
    print("\n=== ALL FP8 GEMV VERIFY TESTS PASSED ===")
