import collections
import queue
import threading
from typing import Optional
from engine.mmap_loader import DirectSSDMmapManager


class AsyncPrefetchEngine:
    """Asynchronous background prefetch thread with a bounded byte-capped cache.

    Streams layer N+1 expert slices off NVMe SSD via mmap into memory while
    layer N executes compute. Capped by max_bytes to avoid unbounded RAM growth.
    """

    def __init__(
        self,
        mmap_mgr: DirectSSDMmapManager,
        max_bytes: int = 512 * 1024 * 1024,
    ):
        self.mmap_mgr = mmap_mgr
        self.max_bytes = max(0, int(max_bytes))
        self.request_queue: queue.Queue = queue.Queue()
        self.cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self.pending: set[str] = set()
        self.current_bytes: int = 0
        self.lock = threading.Lock()
        self.running = True
        self.worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.worker_thread.start()

    def _prefetch_loop(self):
        while self.running:
            try:
                tensor_name, byte_len = self.request_queue.get(timeout=0.1)
                try:
                    data = self.mmap_mgr.fetch_expert_slice_direct(tensor_name, byte_len)
                    if data:
                        with self.lock:
                            data_len = len(data)
                            if tensor_name in self.cache:
                                self.current_bytes -= len(self.cache[tensor_name])
                                del self.cache[tensor_name]

                        # Evict oldest unconsumed entries if exceeding max_bytes
                            while self.cache and self.current_bytes + data_len > self.max_bytes:
                                _, evicted_data = self.cache.popitem(last=False)
                                self.current_bytes -= len(evicted_data)

                            if self.current_bytes + data_len <= self.max_bytes:
                                self.cache[tensor_name] = data
                                self.current_bytes += data_len
                finally:
                    with self.lock:
                        self.pending.discard(tensor_name)
                    self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[Prefetch] Failed to fetch {tensor_name}: {exc}")

    def request_prefetch(self, tensor_name: str, byte_length: int = 2048):
        with self.lock:
            if tensor_name in self.cache or tensor_name in self.pending:
                return
            self.pending.add(tensor_name)
        self.request_queue.put((tensor_name, byte_length))

    def get_prefetched_data(self, tensor_name: str) -> Optional[bytes]:
        with self.lock:
            data = self.cache.pop(tensor_name, None)
            if data is not None:
                self.current_bytes -= len(data)
            return data

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "cached_items": len(self.cache),
                "current_bytes": self.current_bytes,
                "max_bytes": self.max_bytes,
            }

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
