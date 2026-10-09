import threading
import queue
from typing import Dict, Optional
from engine.mmap_loader import DirectSSDMmapManager

class AsyncPrefetchEngine:
    """
    Asynchronous background prefetch thread. Streams layer N+1 expert slices
    off NVMe SSD via mmap into memory while layer N evaluates compute.
    """
    def __init__(self, mmap_mgr: DirectSSDMmapManager):
        self.mmap_mgr = mmap_mgr
        self.request_queue = queue.Queue()
        self.cache: Dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.running = True
        self.worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.worker_thread.start()

    def _prefetch_loop(self):
        while self.running:
            try:
                tensor_name, byte_len = self.request_queue.get(timeout=0.1)
                data = self.mmap_mgr.fetch_expert_slice_direct(tensor_name, byte_len)
                if data:
                    with self.lock:
                        self.cache[tensor_name] = data
                self.request_queue.task_done()
            except queue.Empty:
                continue

    def request_prefetch(self, tensor_name: str, byte_length: int = 2048):
        if tensor_name not in self.cache:
            self.request_queue.put((tensor_name, byte_length))

    def get_prefetched_data(self, tensor_name: str) -> Optional[bytes]:
        with self.lock:
            return self.cache.pop(tensor_name, None)

    def stop(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
