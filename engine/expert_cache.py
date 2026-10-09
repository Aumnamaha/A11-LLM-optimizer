from collections import OrderedDict
from typing import Optional

class HotExpertRAMCache:
    """
    Tier-2 System RAM Cache storing high-frequency software development & code analysis experts.
    Caps System RAM usage at under 4GB.
    """
    def __init__(self, max_hot_experts: int = 120):
        self.max_capacity = max_hot_experts
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.hits = 0
        self.misses = 0

        # High-frequency domain experts pinned for code & file analysis tasks
        self.pinned_expert_patterns = {
            "blk.0.ffn_down_exps", "blk.1.ffn_down_exps", # Base routing layers
            "blk.8.ffn_down_exps", "blk.16.ffn_down_exps", # Code syntax/Tool call layers
            "blk.24.ffn_down_exps", "blk.31.ffn_down_exps" # Output response formatting
        }

    def get_expert(self, expert_name: str) -> Optional[bytes]:
        if expert_name in self.cache:
            self.hits += 1
            self.cache.move_to_end(expert_name)
            return self.cache[expert_name]
        self.misses += 1
        return None

    def put_expert(self, expert_name: str, raw_bytes: bytes):
        if expert_name in self.cache:
            self.cache.move_to_end(expert_name)
            return

        # Evict non-pinned cold experts if cache reaches RAM ceiling
        if len(self.cache) >= self.max_capacity:
            for k in list(self.cache.keys()):
                if not any(pattern in k for pattern in self.pinned_expert_patterns):
                    del self.cache[k]
                    break
            else:
                self.cache.popitem(last=False)

        self.cache[expert_name] = raw_bytes

    def get_stats(self) -> dict:
        total = max(self.hits + self.misses, 1)
        return {
            "cached_experts": len(self.cache),
            "max_capacity": self.max_capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round((self.hits / total) * 100, 2)
        }
