import sys
import os
import warnings
from typing import Optional
import torch
import numpy as np

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import a11_cuda_engine
    HAS_CPP_EXTENSION = True
    print("[Quant Engine] Native C++ compiled extension loaded successfully.")
except ImportError as e:
    HAS_CPP_EXTENSION = False
    print(f"[Quant Engine] C++ module fallback triggered ({e}). Running Python/NumPy execution.")


def dequantize_q2_k_numpy(raw_bytes: bytes, num_elements: Optional[int] = None) -> np.ndarray:
    """Full vectorized NumPy implementation of canonical GGUF Q2_K dequantization.

    A Q2_K block holds 256 weights in 84 bytes:
      - 16 bytes: scales (low 4 bits scale, high 4 bits min)
      - 64 bytes: quants (2-bit values)
      - 2 bytes: super-block scale d (float16)
      - 2 bytes: super-block min dmin (float16)
    """
    if not raw_bytes or (num_elements is not None and num_elements <= 0):
        return np.zeros(num_elements or 0, dtype=np.float32)

    n_blocks = len(raw_bytes) // 84
    if n_blocks == 0:
        return np.zeros(num_elements or 0, dtype=np.float32)

    blocks = np.frombuffer(raw_bytes[: n_blocks * 84], dtype=np.uint8).reshape((n_blocks, 84))

    scales, rest = np.hsplit(blocks, [16])
    qs, rest = np.hsplit(rest, [64])
    d, dmin = np.hsplit(rest, [2])

    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)

    dl = (d * (scales & 0x0F).astype(np.float32)).reshape((n_blocks, 16, 1))
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, 16, 1))

    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)
    qs = qs.reshape((n_blocks, 16, 16)).astype(np.float32)

    out = (dl * qs - ml).reshape((n_blocks * 256,))
    if num_elements is not None:
        if len(out) < num_elements:
            out = np.pad(out, (0, num_elements - len(out)))
        else:
            out = out[:num_elements]
    return out


class GGUFDequantizer:
    @staticmethod
    def dequantize_q2_k(raw_bytes: bytes, num_elements: int) -> torch.Tensor:
        """Dequantize GGUF Q2_K quantized bytes into float32 PyTorch tensor."""
        if num_elements <= 0 or len(raw_bytes) == 0:
            return torch.zeros(max(0, num_elements), dtype=torch.float32)

        if HAS_CPP_EXTENSION:
            try:
                raw_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
                return a11_cuda_engine.dequantize_q2_k_cuda(raw_tensor, num_elements)
            except Exception:
                pass

        arr = dequantize_q2_k_numpy(raw_bytes, num_elements)
        return torch.from_numpy(arr)


if __name__ == "__main__":
    np.random.seed(42)
    test_raw = np.random.randint(0, 256, size=84 * 2, dtype=np.uint8)
    test_raw[80:82] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    test_raw[82:84] = np.frombuffer(np.float16(0.2).tobytes(), dtype=np.uint8)
    test_raw[164:166] = np.frombuffer(np.float16(0.8).tobytes(), dtype=np.uint8)
    test_raw[166:168] = np.frombuffer(np.float16(0.1).tobytes(), dtype=np.uint8)
    result = GGUFDequantizer.dequantize_q2_k(test_raw.tobytes(), 512)
    print(f"[Engine Test] Dequantized {result.shape[0]} elements | Sample: {result[:4].tolist()}")
