import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

from engine.loader import SplitGGUFLoader
from engine.mmap_loader import DirectSSDMmapManager

if TYPE_CHECKING:
    from engine.pager import MemoryPager


def tensor_n_bytes(shape, qtype_int: int) -> int:
    qtype = GGMLQuantizationType(qtype_int)
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_elems = math.prod(shape)
    return (n_elems // block_size) * type_size


def is_expert_tensor(name: str) -> bool:
    """True for pageable MoE expert tensors (`*_exps`)."""
    return "exps" in name.lower()


def estimate_resident_bytes(loader, dtype: torch.dtype = torch.bfloat16) -> int:
    """Bytes the resident core will occupy once dequantized to `dtype`.

    ``WeightStore.get_torch`` dequantizes resident tensors to `dtype` and holds them
    for the process lifetime. Every tensor that is *not* a pageable expert is
    fetched this way, so this is the RAM the expert cache cannot use.
    """
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total = 0
    for name, meta in getattr(loader, "tensors", {}).items():
        if is_expert_tensor(name):
            continue
        total += math.prod(meta["shape"]) * bytes_per_elem
    return total


def plan_ram_budgets(loader, ram_budget_bytes: int, dtype: torch.dtype = torch.bfloat16) -> Tuple[int, int]:
    """Split a RAM budget into (resident_bytes, expert_cache_bytes).

    The expert cache must be sized from what is *left* of the RAM budget after
    the resident core, otherwise resident + cache oversubscribes host RAM.
    """
    resident = estimate_resident_bytes(loader, dtype)
    return resident, max(0, int(ram_budget_bytes) - resident)


def _kind_to_page_tensor(kind: str) -> str:
    """Map internal kind to page-table tensor tag."""
    return kind.upper()  # 'gate'->'GATE', 'up'->'UP', 'down'->'DOWN'


class WeightStore:
    """
    Serves dequantized torch tensors from the mmap'd GGUF shard.

    - get_torch(): resident hot weights (norms, attn, router, embed, output),
      dequantized once and cached unboundedly in `resident_dtype`.
    - get_expert(): per-expert MoE FFN weights, served via MemoryPager.
    """

    def __init__(
        self,
        loader: SplitGGUFLoader,
        mmap_mgr: DirectSSDMmapManager,
        pager: Optional["MemoryPager"] = None,  # type: ignore[name-defined]
        n_experts: int = 128,
        resident_dtype: torch.dtype = torch.bfloat16,
    ):
        # Late import to avoid circular dependency
        from engine.pager import MemoryPager as _MemoryPager
        MemoryPager = _MemoryPager

        self.loader = loader
        self.mmap_mgr = mmap_mgr
        self.n_experts = n_experts
        self.pager = pager
        self.resident_dtype = resident_dtype

        self._resident: Dict[str, torch.Tensor] = {}
        self._resident_bytes = 0

    def _read_raw(self, name: str, byte_offset: int, n_bytes: int) -> np.ndarray:
        meta = self.loader.tensors[name]
        shard_key = meta.get("shard_path") or meta.get("file_path") or meta.get("shard")
        mm = self.mmap_mgr.get_mmap(shard_key)
        if mm is None:
            raise ValueError(f"No mmap handle available for shard '{shard_key}' of tensor '{name}'")
        offset = meta.get("offset", 0)
        start = offset + byte_offset
        buf = mm[start : start + n_bytes]
        if len(buf) != n_bytes:
            raise IOError(
                f"Short read for tensor '{name}': expected {n_bytes} bytes, got {len(buf)}"
            )
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
        # Store in resident_dtype (bfloat16 by default)
        np_dtype = np.float16 if self.resident_dtype == torch.float16 else np.float32
        if self.resident_dtype == torch.bfloat16:
            np_dtype = np.float32  # numpy doesn't have bfloat16, use float32 as intermediate
        t = torch.from_numpy(np.array(arr, dtype=np_dtype, copy=True))
        if self.resident_dtype == torch.bfloat16:
            t = t.to(torch.bfloat16)
        elif self.resident_dtype == torch.float16:
            t = t.to(torch.float16)
        self._resident[name] = t
        self._resident_bytes += t.numel() * t.element_size()
        return t

    def _page_id_for_expert(self, layer: int, kind: str, expert_idx: int) -> str:
        """Construct the logical page ID for a single-page expert."""
        tensor_tag = _kind_to_page_tensor(kind)
        return f"E{expert_idx:03d}:L{layer:02d}:T{tensor_tag}:P000"

    def get_expert(self, layer: int, kind: str, expert_idx: int) -> torch.Tensor:
        """kind: 'gate' | 'up' | 'down'. Returns [out_features, in_features] fp32."""
        if self.pager is not None:
            page_id = self._page_id_for_expert(layer, kind, expert_idx)
            return self.pager.get_page(page_id)

        # Fallback (should not be used once pager is wired)
        key = (layer, kind, expert_idx)
        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        meta = self.loader.tensors[name]
        shape = meta["shape"]            # [in, out, n_experts]
        qtype_int = int(meta["type"])
        per_expert = tensor_n_bytes(shape, qtype_int) // self.n_experts

        raw = self._read_raw(name, expert_idx * per_expert, per_expert)
        arr = self._dequant(raw, qtype_int, shape[:2])
        return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))

    @property
    def resident_bytes(self) -> int:
        """Actual bytes currently held for resident tensors."""
        return self._resident_bytes