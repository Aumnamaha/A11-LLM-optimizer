"""Asynchronous prefetcher for expert pages.

Replaces the legacy raw-byte prefetcher with a pager-aware implementation.
The background worker pulls logical page IDs from a queue and calls
MemoryPager.get_page() to trigger the full SSD → RAM → VRAM pipeline.
"""
import queue
import threading
from typing import Optional, Set

import torch

from engine.pager import MemoryPager


class AsyncPrefetcher:
    """Background worker that preloads expert pages into the memory hierarchy.

    The main thread dispatches logical page IDs (e.g., "E005:L12:TGATE:P000")
    immediately after the MoE router selects experts. The background worker
    calls MemoryPager.get_page() to trigger the full SSD → RAM → VRAM pipeline,
    so the tensor is resident when the forward pass actually needs it.
    """

    def __init__(
        self,
        pager: MemoryPager,
        num_workers: int = 1,
    ):
        self.pager = pager
        self.num_workers = max(1, num_workers)
        self.request_queue: "queue.Queue[str]" = queue.Queue()
        self.pending: Set[str] = set()
        self._lock = threading.Lock()
        self.running = True
        self.workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"prefetch-{i}", daemon=True)
            t.start()
            self.workers.append(t)

    def _worker_loop(self):
        while self.running:
            try:
                page_id = self.request_queue.get(timeout=0.1)
                try:
                    # This triggers the full SSD → RAM → VRAM pipeline
                    # and populates the pager's caches.
                    _ = self.pager.get_page(page_id)
                except Exception as exc:
                    print(f"[Prefetch] Failed to fetch {page_id}: {exc}")
                finally:
                    with self._lock:
                        self.pending.discard(page_id)
                    self.request_queue.task_done()
            except queue.Empty:
                continue

    def request_prefetch(self, page_id: str) -> None:
        """Enqueue a logical page ID for background loading."""
        with self._lock:
            if page_id in self.pending:
                return
            self.pending.add(page_id)
        self.request_queue.put(page_id)

    def request_prefetch_batch(self, page_ids: list[str]) -> None:
        """Enqueue multiple page IDs at once."""
        with self._lock:
            for pid in page_ids:
                if pid in self.pending:
                    continue
                self.pending.add(pid)
            for pid in page_ids:
                self.request_queue.put(pid)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "queue_size": self.request_queue.qsize(),
                "pending_count": len(self.pending),
                "workers": self.num_workers,
            }

    def stop(self):
        self.running = False
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=1.0)