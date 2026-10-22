from collections import OrderedDict
from typing import Optional


class HotExpertRAMCache:
    """Tier-2 System RAM Cache storing active MoE experts.

    Supports byte-based memory budgets to prevent OOM on memory-constrained systems,
    while maintaining LRU eviction and pattern pinning for high-frequency layers.
    """

    def __init__(
        self,
        max_bytes: Optional[int] = None,
        max_hot_experts: Optional[int] = None,
    ):
        # Default to 4GB if neither limit is specified
        if max_bytes is None and max_hot_experts is None:
            self.max_bytes = 4 * 1024 * 1024 * 1024
        else:
            self.max_bytes = max_bytes

        self.max_hot_experts = max_hot_experts
        self.max_capacity = max_hot_experts  # Backward compatibility
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.current_bytes: int = 0
        self.hits = 0
        self.misses = 0

        # High-frequency domain experts pinned for code & file analysis tasks
        self.pinned_expert_patterns = {
            "blk.0.ffn_down_exps", "blk.1.ffn_down_exps",  # Base routing layers
            "blk.8.ffn_down_exps", "blk.16.ffn_down_exps",  # Code syntax/Tool call layers
            "blk.24.ffn_down_exps", "blk.31.ffn_down_exps"  # Output response formatting
        }

    def get_expert(self, expert_name: str) -> Optional[bytes]:
        if expert_name in self.cache:
            self.hits += 1
            self.cache.move_to_end(expert_name)
            return self.cache[expert_name]
        self.misses += 1
        return None

    def put_expert(self, expert_name: str, raw_bytes: bytes):
        item_bytes = len(raw_bytes)
        if expert_name in self.cache:
            self.current_bytes -= len(self.cache[expert_name])
            del self.cache[expert_name]

        # Evict entries if either byte budget or entry budget is exceeded
        while self.cache and (
            (self.max_bytes is not None and self.current_bytes + item_bytes > self.max_bytes)
            or (self.max_hot_experts is not None and len(self.cache) >= self.max_hot_experts)
        ):
            # Prioritize evicting non-pinned entries first in LRU order
            evicted_key = None
            for k in self.cache:
                if not any(pattern in k for pattern in self.pinned_expert_patterns):
                    evicted_key = k
                    break
            if evicted_key is None:
                # If all entries are pinned, evict oldest pinned entry
                evicted_key = next(iter(self.cache))

            self.current_bytes -= len(self.cache[evicted_key])
            del self.cache[evicted_key]

        if self.max_bytes is None or self.current_bytes + item_bytes <= self.max_bytes:
            self.cache[expert_name] = raw_bytes
            self.current_bytes += item_bytes

    def get_stats(self) -> dict:
        total = max(self.hits + self.misses, 1)
        return {
            "cached_experts": len(self.cache),
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "max_capacity": self.max_hot_experts if self.max_hot_experts is not None else len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round((self.hits / total) * 100, 2),
        }

    def clear(self):
        self.cache.clear()
        self.current_bytes = 0
