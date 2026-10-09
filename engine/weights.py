import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

from engine.loader import SplitGGUFLoader
from engine.mmap_loader import DirectSSDMmapManager


def tensor_n_bytes(shape, qtype_int: int) -> int:
    qtype = GGMLQuantizationType(qtype_int)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_elems = math.prod(shape)
    return (n_elems // block_size) * type_size


class WeightStore:
    """
    Serves dequantized torch tensors from the mmap'd GGUF shard.

    - get_torch(): resident hot weights (norms, attn, router, embed, output),
      dequantized once and cached unboundedly (small total footprint).
    - get_expert(): per-expert MoE FFN weights, LRU-capped by a byte budget.
    """

    def __init__(self, loader: SplitGGUFLoader, mmap_mgr: DirectSSDMmapManager,
                 expert_cache_bytes: int = 3 * 1024 ** 3, n_experts: int = 128):
        self.loader = loader
        self.mmap_mgr = mmap_mgr
        self.n_experts = n_experts
        self.expert_cache_budget = expert_cache_bytes

        self._resident: Dict[str, torch.Tensor] = {}
        self._expert_lru: OrderedDict[Tuple[int, str, int], torch.Tensor] = OrderedDict()
        self._expert_lru_bytes = 0

        # mmap handle for the (single) shard
        self._mm = self.mmap_mgr.mmaps.get(self.loader.shards[0])

    def _read_raw(self, name: str, byte_offset: int, n_bytes: int) -> np.ndarray:
        meta = self.loader.tensors[name]
        self._mm.seek(meta["offset"] + byte_offset)
        buf = self._mm.read(n_bytes)
        return np.frombuffer(buf, dtype=np.uint8)

    def _dequant(self, raw: np.ndarray, qtype_int: int, shape) -> np.ndarray:
        # GGUF/ggml stores tensors with ne0 (the first dim) contiguous. A flat
        # memory-order buffer therefore maps to reversed(dims) in C order, which
        # is also the PyTorch-native weight layout: [out_features, in_features].
        qtype = GGMLQuantizationType(qtype_int)
        if qtype == GGMLQuantizationType.F32:
            flat = raw.view(np.float32)
        else:
            flat = dequantize(raw, qtype)
        return flat.reshape(tuple(reversed(shape)))

    def get_torch(self, name: str) -> torch.Tensor:
        t = self._resident.get(name)
        if t is not None:
            return t
        meta = self.loader.tensors[name]
        shape = meta["shape"]
        qtype_int = int(meta["type"])
        raw = self._read_raw(name, 0, tensor_n_bytes(shape, qtype_int))
        arr = self._dequant(raw, qtype_int, shape)
        # copy() -> writable, non-mmapped tensor (silences torch.from_numpy warning)
        t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        self._resident[name] = t
        return t

    def expert_byte_size(self, name: str) -> int:
        meta = self.loader.tensors[name]
        return tensor_n_bytes(meta["shape"], int(meta["type"])) // self.n_experts

    def get_expert(self, layer: int, kind: str, expert_idx: int) -> torch.Tensor:
        """kind: 'gate' | 'up' | 'down'. Returns [out_features, in_features] fp32."""
        key = (layer, kind, expert_idx)
        t = self._expert_lru.get(key)
        if t is not None:
            self._expert_lru.move_to_end(key)
            return t

        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        meta = self.loader.tensors[name]
        shape = meta["shape"]            # [in, out, n_experts]
        qtype_int = int(meta["type"])
        per_expert = tensor_n_bytes(shape, qtype_int) // self.n_experts

        raw = self._read_raw(name, expert_idx * per_expert, per_expert)
        arr = self._dequant(raw, qtype_int, shape[:2])
        t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))

        t_bytes = t.numel() * 4
        while self._expert_lru_bytes + t_bytes > self.expert_cache_budget and self._expert_lru:
            _, evicted = self._expert_lru.popitem(last=False)
            self._expert_lru_bytes -= evicted.numel() * 4
        self._expert_lru[key] = t
        self._expert_lru_bytes += t_bytes
        return t

    @property
    def expert_cache_stats(self) -> Tuple[int, int]:
        return len(self._expert_lru), self._expert_lru_bytes
