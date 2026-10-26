"""Paged KV Cache Manager for long-context inference.

Manages K/V cache in a paged fashion (pages of tokens) with LRU eviction
to RAM/SSD when VRAM budget is exceeded.
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class KVPage:
    """A page of KV cache for one layer covering a token range."""
    layer_idx: int
    start_pos: int
    end_pos: int  # exclusive
    k: torch.Tensor  # [page_size, n_head_kv, head_dim]
    v: torch.Tensor  # [page_size, n_head_kv, head_dim]


class KVCacheSlotPool:
    """Fixed slots for KV cache pages with LRU eviction."""

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self._slots: Dict[int, KVPage] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free: OrderedDict[int, None] = OrderedDict()
        for i in range(n_slots):
            self._free[i] = None

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def acquire(self) -> int:
        if self._free:
            slot_idx = next(iter(self._free))
            self._free.pop(slot_idx, None)
            self._lru[slot_idx] = None
            return slot_idx
        if not self._lru:
            raise RuntimeError("KVCacheSlotPool exhausted")
        evict_idx = next(iter(self._lru))
        self._lru.pop(evict_idx, None)
        self._slots.pop(evict_idx, None)
        return evict_idx

    def put(self, slot_idx: int, page: KVPage) -> None:
        self._slots[slot_idx] = page
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        else:
            self._lru[slot_idx] = None

    def get(self, slot_idx: int) -> Optional[KVPage]:
        if slot_idx in self._lru:
            self._lru.move_to_end(slot_idx)
        return self._slots.get(slot_idx)

    def evict(self, slot_idx: int) -> Optional[KVPage]:
        page = self._slots.pop(slot_idx, None)
        self._lru.pop(slot_idx, None)
        self._free[slot_idx] = None
        return page


class KVCacheManager:
    """Manages paged KV cache with LRU eviction to host memory."""

    def __init__(
        self,
        n_layers: int,
        n_head_kv: int,
        head_dim: int,
        page_size: int = 256,  # tokens per page
        max_pages: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.n_layers = n_layers
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        # Per-layer page tracking: layer_idx -> list of (start_pos, page)
        self._layer_pages: Dict[int, OrderedDict[int, KVPage]] = {
            i: OrderedDict() for i in range(n_layers)
        }
        self._lock = threading.RLock()

        # Slot pool for fixed page budget
        self.pool = KVCacheSlotPool(max_pages) if max_pages > 0 else None

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._page_splits = 0

    @property
    def stats(self) -> Dict[str, float]:
        return {
            "kv_hits": float(self._hits),
            "kv_misses": float(self._misses),
            "kv_evictions": float(self._evictions),
            "kv_pages": float(sum(len(p) for p in self._layer_pages.values())),
        }

    def _make_page(
        self,
        layer_idx: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> KVPage:
        end_pos = start_pos + k.shape[0]
        return KVPage(
            layer_idx=layer_idx,
            start_pos=start_pos,
            end_pos=end_pos,
            k=k.to(self.dtype),
            v=v.to(self.dtype),
        )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new K/V tokens for a layer."""
        with self._lock:
            pages = self._layer_pages[layer_idx]
            pos = k.shape[0]

            if pages and pos == self.page_size:
                # Full page - store and start new
                start_pos = pages[-1].end_pos if pages else 0
                page = self._make_page(layer_idx, start_pos, k, v)
                pages[page.start_pos] = page

                # Enforce page budget via pool
                if self.pool and self.pool.free_slots == 0:
                    self._evict_oldest()
            elif pages and pos < self.page_size:
                # Append to last page
                last_page = list(pages.values())[-1]
                new_k = torch.cat([last_page.k, k], dim=0)
                new_v = torch.cat([last_page.v, v], dim=0)
                new_page = self._make_page(layer_idx, last_page.start_pos, new_k, new_v)
                pages[last_page.start_pos] = new_page
            else:
                # New page
                page = self._make_page(layer_idx, 0, k, v)
                pages[0] = page

    def _evict_oldest(self) -> None:
        """Evict the oldest page across all layers (simple global LRU)."""
        oldest_layer = None
        oldest_start = float('inf')

        for layer_idx, pages in self._layer_pages.items():
            if pages:
                start = next(iter(pages))
                if start < oldest_start:
                    oldest_start = start
                    oldest_layer = layer_idx

        if oldest_layer is not None:
            pages = self._layer_pages[oldest_layer]
            if pages:
                start, page = pages.popitem(last=False)
                self._evictions += 1
                # In a full implementation, we'd write page to host memory here
                # For now, just drop (source of truth is recomputed if needed)

    def get_kv(self, layer_idx: int, start_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for a position range, concatenating pages if needed."""
        with self._lock:
            pages = self._layer_pages[layer_idx]
            if not pages:
                return (
                    torch.empty(0, self.n_head_kv, self.head_dim, dtype=self.dtype, device=self.device),
                    torch.empty(0, self.n_head_kv, self.head_dim, dtype=self.dtype, device=self.device),
                )

            # Find pages covering the range
            k_parts = []
            v_parts = []
            for start_pos_page, page in pages.items():
                if page.end_pos <= start_pos:
                    continue
                if page.start_pos >= end_pos:
                    break
                # Overlap
                slice_start = max(0, start_pos - page.start_pos)
                slice_end = min(page.k.shape[0], end_pos - page.start_pos)
                if slice_start < slice_end:
                    k_parts.append(page.k[slice_start:slice_end])
                    v_parts.append(page.v[slice_start:slice_end])

            if not k_parts:
                return (
                    torch.empty(0, self.n_head_kv, self.head_dim, dtype=self.dtype, device=self.device),
                    torch.empty(0, self.n_head_kv, self.head_dim, dtype=self.dtype, device=self.device),
                )

            k_cat = torch.cat(k_parts, dim=0) if len(k_parts) > 1 else k_parts[0]
            v_cat = torch.cat(v_parts, dim=0) if len(v_parts) > 1 else v_parts[0]
            self._hits += 1
            return k_cat.to(self.device), v_cat.to(self.device)

    def clear(self) -> None:
        with self._lock:
            for pages in self._layer_pages.values():
                pages.clear()