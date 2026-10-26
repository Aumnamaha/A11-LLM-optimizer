import os
import mmap
from typing import Dict, Optional
from engine.loader import SplitGGUFLoader

class DirectSSDMmapManager:
    """
    Zero-Copy NVMe SSD mmap engine. Keeps file handles open for DMA transfers
    directly to GPU VRAM without loading weights into System RAM.
    """
    def __init__(self, loader: SplitGGUFLoader):
        self.loader = loader
        self.mmaps: Dict[str, mmap.mmap] = {}
        self.file_handles = []
        self._initialize_mmaps()

    def _initialize_mmaps(self):
        for shard_path in self.loader.shards:
            try:
                f = open(shard_path, "rb")
                self.file_handles.append(f)
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.mmaps[shard_path] = mm
                self.mmaps[os.path.basename(shard_path)] = mm
            except Exception as e:
                print(f"[mmap Manager] Error mapping {shard_path}: {e}")

    def get_mmap(self, shard_key: Optional[str] = None) -> Optional[mmap.mmap]:
        """Retrieve mmap object using full shard path, basename, or default."""
        if not shard_key:
            if self.loader.shards and self.loader.shards[0] in self.mmaps:
                return self.mmaps[self.loader.shards[0]]
            return None
        if shard_key in self.mmaps:
            return self.mmaps[shard_key]
        base = os.path.basename(shard_key)
        if base in self.mmaps:
            return self.mmaps[base]
        for s in self.loader.shards:
            if s.endswith(shard_key) or os.path.basename(s) == base:
                return self.mmaps.get(s)
        return None

    def _get_tensor_metadata(self, tensor_name: str) -> Optional[dict]:
        """Resolve tensor metadata across loader index attributes."""
        if hasattr(self.loader, 'tensor_map') and tensor_name in self.loader.tensor_map:
            return self.loader.tensor_map[tensor_name]
        if hasattr(self.loader, 'tensors') and tensor_name in self.loader.tensors:
            return self.loader.tensors[tensor_name]
        if hasattr(self.loader, 'tensor_index') and tensor_name in self.loader.tensor_index:
            return self.loader.tensor_index[tensor_name]
        
        # Partial match fallback for sharded GGUF expert tensor keys
        tensor_dict = getattr(self.loader, 'tensors', getattr(self.loader, 'tensor_map', getattr(self.loader, 'tensor_index', {})))
        for k, v in tensor_dict.items():
            if tensor_name in k:
                return v
        return None

    def fetch_expert_slice_direct(self, tensor_name: str, max_bytes: int = 4096) -> Optional[bytes]:
        """Fetch targeted expert slice directly from NVMe SSD virtual memory."""
        meta = self._get_tensor_metadata(tensor_name)
        if not meta:
            return None

        # Resolve shard path and file offset from tensor metadata
        shard_key = meta.get("shard_path") or meta.get("file_path") or meta.get("shard")
        offset = meta.get("offset", 0)
        size = meta.get("size", meta.get("n_bytes", max_bytes))

        mm = self.get_mmap(shard_key)
        if mm:
            try:
                return mm[offset : offset + min(size, max_bytes)]
            except Exception:
                return None
        return None

    def fetch_tensor_bytes(self, tensor_name: str, max_bytes: int = 4096) -> Optional[bytes]:
        return self.fetch_expert_slice_direct(tensor_name, max_bytes)

    def close(self):
        for mm in set(self.mmaps.values()):
            try:
                mm.close()
            except Exception:
                pass
        self.mmaps.clear()
        for f in self.file_handles:
            try:
                f.close()
            except Exception:
                pass
        self.file_handles.clear()
        print("[mmap Manager] Memory maps closed successfully.")

# Alias for backward compatibility across engine modules
MemoryMappedTensorManager = DirectSSDMmapManager
