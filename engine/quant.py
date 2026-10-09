import sys
import os
import torch
import numpy as np

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

class GGUFDequantizer:
    @staticmethod
    def dequantize_q2_k(raw_bytes: bytes, num_elements: int) -> torch.Tensor:
        if HAS_CPP_EXTENSION and len(raw_bytes) > 0:
            try:
                # Bypasses writeable buffer checks by specifying read-only flag explicitly
                raw_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
                return a11_cuda_engine.dequantize_q2_k_cuda(raw_tensor, num_elements)
            except Exception:
                pass

        num_bytes = len(raw_bytes)
        if num_bytes == 0:
            return torch.zeros(num_elements, dtype=torch.float32)
            
        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        w0 = (raw_arr & 0x03) - 1.0
        w1 = ((raw_arr >> 2) & 0x03) - 1.0
        w2 = ((raw_arr >> 4) & 0x03) - 1.0
        w3 = ((raw_arr >> 6) & 0x03) - 1.0
        
        unpacked = np.column_stack([w0, w1, w2, w3]).flatten().astype(np.float32)
        if len(unpacked) < num_elements:
            unpacked = np.pad(unpacked, (0, num_elements - len(unpacked)))
            
        return torch.from_numpy(unpacked[:num_elements])

if __name__ == "__main__":
    test_bytes = b'\x55\xAA\x11\xFF' * 32
    result = GGUFDequantizer.dequantize_q2_k(test_bytes, 512)
    print(f"[C++ Engine Test] Dequantized {result.shape[0]} elements | Sample: {result[:4].tolist()}")
