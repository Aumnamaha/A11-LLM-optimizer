import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize

from engine.loader import SplitGGUFLoader
from engine.mmap_loader import DirectSSDMmapManager


@dataclass
class PageTableEntry:
    id: str
    layer: int
    expert: int
    tensor: str  # 'gate','up','down'
    page: int
    pages_per_expert: int
    byte_offset: int  # absolute file offset
    byte_length: int
    shard: str
    cache_slot_ram: int = -1      # RAM tier slot index
    cache_slot_vram: int = -1     # VRAM tier slot index


class ModelPageTable:
    """Loads and indexes the logical page table (JSONL)."""

    def __init__(self, path: str, loader: Optional[SplitGGUFLoader] = None):
        self.path = path
        self.entries: Dict[str, PageTableEntry] = {}
        self.by_tensor_expert_page: Dict[Tuple[str, int, int], str] = {}
        self._load()

    def _load(self) -> None:
        import json

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                pe = PageTableEntry(
                    id=e["id"],
                    layer=int(e["layer"]),
                    expert=int(e["expert"]),
                    tensor=e["tensor"],
                    page=int(e["page"]),
                    pages_per_expert=int(e["pages_per_expert"]),
                    byte_offset=int(e["byte_offset"]),
                    byte_length=int(e["byte_length"]),
                    shard=e.get("shard", ""),
                )
                self.entries[pe.id] = pe
                tensor_name = f"blk.{pe.layer}.ffn_{pe.tensor}_exps.weight"
                self.by_tensor_expert_page[(tensor_name, pe.expert, pe.page)] = pe.id

    def get(self, logical_id: str) -> Optional[PageTableEntry]:
        return self.entries.get(logical_id)

    def get_page_id_for_expert_page(self, tensor_name: str, expert_idx: int, page_idx: int) -> Optional[str]:
        return self.by_tensor_expert_page.get((tensor_name, expert_idx, page_idx))


class RAMSlotPool:
    """Fixed number of RAM slots with LRU eviction."""

    def __init__(self, n_slots: int):
        if n_slots < 0:
            n_slots = 0
        self.n_slots = n_slots
        self._slots: Dict[int, Union[torch.Tensor, bytes, np.ndarray]] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free: OrderedDict[int, None] = OrderedDict()
        for i in range(n_slots):
            self._free[i] = None
        self.evictions = 0

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def acquire(self) -> int:
        """Acquire an available slot; if none free, evict least-recently used."""
        if self._free:
            slot_idx = next(iter(self._free))
            self._free.pop(slot_idx, None)
            self._lru[slot_idx] = None
            return slot_idx
        if not self._lru:
            raise RuntimeError("RAMSlotPool has zero slots and no evictable slots")
        # evict LRU
        evict_idx = next(iter(self._lru))
        self._lru.pop(evict_idx, None)
        self._slots.pop(evict_idx, None)
        self.evictions += 1
        return evict_idx

    def release(self, slot_idx: int) -> None:
        self._slots.pop(slot_idx, None)
        self._lru.pop(slot_idx, None)
        self._free[slot_idx] = None

    def put(self, slot_idx: int, payload: Union[torch.Tensor, bytes, np.ndarray]) -> None:
        self._slots[slot_idx] = payload
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        else:
            self._lru[slot_idx] = None

    def get(self, slot_idx: int) -> Optional[Union[torch.Tensor, bytes, np.ndarray]]:
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        return self._slots.get(slot_idx)

    def touch(self, slot_idx: int) -> None:
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)


class VRAMSlotPool:
    """Fixed number of VRAM slots with LRU eviction.
    
    When evicting, GPU tensors are simply deleted (they exist in RAM tier).
    """

    def __init__(self, n_slots: int, device: torch.device):
        if n_slots < 0:
            n_slots = 0
        self.n_slots = n_slots
        self.device = device
        self._slots: Dict[int, torch.Tensor] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free: OrderedDict[int, None] = OrderedDict()
        for i in range(n_slots):
            self._free[i] = None
        self.evictions = 0

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def acquire(self) -> int:
        """Acquire an available VRAM slot; if none free, evict LRU (drop GPU tensor)."""
        if self._free:
            slot_idx = next(iter(self._free))
            self._free.pop(slot_idx, None)
            self._lru[slot_idx] = None
            return slot_idx
        if not self._lru:
            raise RuntimeError("VRAMSlotPool has zero slots and no evictable slots")
        # evict LRU - simply drop the GPU tensor (source of truth is RAM tier)
        evict_idx = next(iter(self._lru))
        self._lru.pop(evict_idx, None)
        self._slots.pop(evict_idx, None)
        self.evictions += 1
        return evict_idx

    def put(self, slot_idx: int, tensor: torch.Tensor) -> None:
        self._slots[slot_idx] = tensor
        self._free.pop(slot_idx, None)  # remove from free list
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        else:
            self._lru[slot_idx] = None

    def get(self, slot_idx: int) -> Optional[torch.Tensor]:
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        return self._slots.get(slot_idx)

    def touch(self, slot_idx: int) -> None:
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)

    def clear(self) -> None:
        """Clear all VRAM tensors."""
        self._slots.clear()
        self._lru.clear()
        self._free = OrderedDict((i, None) for i in range(self.n_slots))


class MemoryPager:
    """Page-level memory controller for non-resident weights (experts).
    
    Implements 3-tier hierarchy: SSD (Tier 3) → System RAM (Tier 2) → VRAM (Tier 1).
    """

    def __init__(
        self,
        page_table: ModelPageTable,
        loader: SplitGGUFLoader,
        mmap_mgr: DirectSSDMmapManager,
        n_slots_ram: int = 0,
        n_slots_vram: int = 0,
        vram_device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        self.page_table = page_table
        self.loader = loader
        self.mmap_mgr = mmap_mgr
        self.vram_device = vram_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vram_enabled = n_slots_vram > 0 and self.vram_device.type == "cuda"
        self.compute_dtype = compute_dtype
        
        # Tier 2: System RAM cache
        self.ram_pool = RAMSlotPool(n_slots_ram)
        # Tier 1: VRAM cache (only if enabled)
        self.vram_pool = VRAMSlotPool(n_slots_vram, self.vram_device) if self._vram_enabled else None
        
        # Maps logical page ID -> RAM slot index / VRAM slot index
        self._entry_to_ram_slot: Dict[str, int] = {}
        self._slot_to_entry_ram: Dict[int, str] = {}
        self._entry_to_vram_slot: Dict[str, int] = {}
        self._slot_to_entry_vram: Dict[int, str] = {}
        
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._decomp_time_ms = 0.0
        self._faults = 0
        self._pcie_transfers = 0
        self._vram_hits = 0
        self._vram_evictions = 0

    @property
    def stats(self) -> Dict[str, float]:
        vram_evict = float(self.vram_pool.evictions) if self.vram_pool else 0.0
        return {
            "ssd_page_faults": float(self._faults),
            "ram_hits": float(self._hits),
            "ram_evictions": float(self.ram_pool.evictions),
            "vram_hits": float(self._vram_hits),
            "vram_evictions": vram_evict,
            "pcie_transfers": float(self._pcie_transfers),
            "decompression_time_ms": float(self._decomp_time_ms),
        }

    def reset_stats(self) -> None:
        self._hits = self._misses = self._faults = 0
        self._vram_hits = self._vram_evictions = self._pcie_transfers = 0
        self._decomp_time_ms = 0.0
        self.ram_pool.evictions = 0
        self.vram_pool.evictions = 0

    def _read_raw_page(self, entry: PageTableEntry) -> np.ndarray:
        tensor_name = f"blk.{entry.layer}.ffn_{entry.tensor}_exps.weight"
        meta = self.loader.tensors.get(tensor_name)
        if meta is None:
            # fallback search
            for k, v in self.loader.tensors.items():
                if k == tensor_name:
                    meta = v
                    break
        if meta is None:
            raise ValueError(f"Cannot resolve tensor metadata for {tensor_name}")
        shard_key = meta.get("shard_path") or meta.get("file_path") or meta.get("shard") or entry.shard
        mm = self.mmap_mgr.get_mmap(shard_key)
        if mm is None:
            raise ValueError(f"No mmap handle for shard '{shard_key}'")
        # page_table byte_offset is absolute (analyzer: tensor_offset + ...)
        start = entry.byte_offset
        buf = mm[start : start + entry.byte_length]
        if len(buf) != entry.byte_length:
            raise IOError(f"Short read for page {entry.id}: expected {entry.byte_length} got {len(buf)}")
        return np.frombuffer(buf, dtype=np.uint8)

    def _dequant_page(self, raw: np.ndarray, entry: PageTableEntry) -> torch.Tensor:
        tensor_name = f"blk.{entry.layer}.ffn_{entry.tensor}_exps.weight"
        meta = self.loader.tensors.get(tensor_name)
        if meta is None:
            for k, v in self.loader.tensors.items():
                if k == tensor_name:
                    meta = v
                    break
        if meta is None:
            raise ValueError(f"Cannot resolve tensor metadata for {tensor_name}")
        shape = meta["shape"]
        qtype_int = int(meta["type"])
        qtype = GGMLQuantizationType(qtype_int)
        t0 = time.perf_counter()
        if qtype == GGMLQuantizationType.F32:
            flat = raw.view(np.float32)
        else:
            flat = dequantize(raw, qtype)
        # reshape to expert shape [out, in] = reversed(shape[:2])
        arr = flat.reshape(tuple(reversed(shape[:2])))
        # Output in compute_dtype (bfloat16 by default)
        if self.compute_dtype == torch.bfloat16:
            t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(torch.bfloat16)
        elif self.compute_dtype == torch.float16:
            t = torch.from_numpy(np.array(arr, dtype=np.float16, copy=True))
        else:
            t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        self._decomp_time_ms += (time.perf_counter() - t0) * 1000.0
        return t

    def _ensure_ram(self, entry: PageTableEntry) -> torch.Tensor:
        """Ensure page is in RAM tier; fetch from SSD if needed."""
        if entry.id in self._entry_to_ram_slot:
            slot_idx = self._entry_to_ram_slot[entry.id]
            payload = self.ram_pool.get(slot_idx)
            if payload is not None:
                self._hits += 1
                if isinstance(payload, torch.Tensor):
                    return payload
                # if stored as raw bytes, reprocess
                raw = payload if isinstance(payload, np.ndarray) else np.frombuffer(payload, dtype=np.uint8) if isinstance(payload, bytes) else payload
                t = self._dequant_page(raw, entry)
                self.ram_pool.put(slot_idx, t)
                return t
        # RAM miss - fetch from SSD
        self._faults += 1
        raw = self._read_raw_page(entry)
        if entry.pages_per_expert == 1:
            t = self._dequant_page(raw, entry)
            slot_idx = self.ram_pool.acquire()
            # Evict old mapping if slot was reused
            old_id = self._slot_to_entry_ram.pop(slot_idx, None)
            if old_id is not None:
                self._entry_to_ram_slot.pop(old_id, None)
            self._entry_to_ram_slot[entry.id] = slot_idx
            self._slot_to_entry_ram[slot_idx] = entry.id
            self.ram_pool.put(slot_idx, t)
            return t
        else:
            # multi-page: store raw bytes
            slot_idx = self.ram_pool.acquire()
            old_id = self._slot_to_entry_ram.pop(slot_idx, None)
            if old_id is not None:
                self._entry_to_ram_slot.pop(old_id, None)
            self._entry_to_ram_slot[entry.id] = slot_idx
            self._slot_to_entry_ram[slot_idx] = entry.id
            self.ram_pool.put(slot_idx, raw.tobytes())
            t = self._dequant_page(raw, entry)
            return t

    def _ensure_vram(self, entry: PageTableEntry, ram_tensor: torch.Tensor) -> torch.Tensor:
        """Ensure page is in VRAM tier; transfer from RAM if needed."""
        # If VRAM is disabled, return RAM tensor directly
        if not self._vram_enabled:
            return ram_tensor
            
        # Check VRAM hit
        if entry.id in self._entry_to_vram_slot:
            vram_slot = self._entry_to_vram_slot[entry.id]
            vram_tensor = self.vram_pool.get(vram_slot)
            if vram_tensor is not None:
                self._vram_hits += 1
                return vram_tensor
        
        # VRAM miss - transfer from RAM to VRAM
        t0 = time.perf_counter()
        vram_tensor = ram_tensor.to(self.vram_device, non_blocking=False)
        self._pcie_transfers += 1
        
        # Acquire VRAM slot
        vram_slot = self.vram_pool.acquire()
        # Evict old VRAM mapping if any
        old_id = self._slot_to_entry_vram.pop(vram_slot, None)
        if old_id is not None:
            self._entry_to_vram_slot.pop(old_id, None)
        self._entry_to_vram_slot[entry.id] = vram_slot
        self._slot_to_entry_vram[vram_slot] = entry.id
        self.vram_pool.put(vram_slot, vram_tensor)
        
        return vram_tensor

    def get_page(self, logical_id: str) -> torch.Tensor:
        """Get page tensor, ensuring it's in VRAM for compute.
        
        Returns tensor on the VRAM device (GPU) if VRAM enabled, else CPU tensor.
        """
        with self._lock:
            entry = self.page_table.get(logical_id)
            if entry is None:
                raise KeyError(f"Unknown page {logical_id}")
            
            # Step 1: Ensure in RAM tier (fetch from SSD if needed)
            ram_tensor = self._ensure_ram(entry)
            
            # Step 2: Ensure in VRAM tier (transfer from RAM if needed)
            vram_tensor = self._ensure_vram(entry, ram_tensor)
            
            return vram_tensor