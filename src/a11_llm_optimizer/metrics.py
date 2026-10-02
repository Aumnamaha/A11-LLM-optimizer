"""Runtime metrics for request latency and adapter usage."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any


class RequestMetrics:
    """Aggregates request metrics and retains only a bounded prompt-free history."""

    def __init__(self, max_events: int = 1000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.successes = 0
        self.failures = 0
        self.by_adapter: Counter[str] = Counter()

    def record_request(
        self,
        prompt: str,
        adapter: str,
        latency_ms: float,
        success: bool = True,
    ) -> dict[str, Any]:
        del prompt
        latency = float(latency_ms)
        event: dict[str, Any] = {
            "adapter": adapter,
            "latency_ms": latency,
            "success": bool(success),
        }
        self.events.append(event)
        self.total_requests += 1
        self.total_latency_ms += latency
        self.by_adapter[adapter] += 1
        if event["success"]:
            self.successes += 1
        else:
            self.failures += 1
        return dict(event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": round(self.total_latency_ms / self.total_requests, 3)
            if self.total_requests
            else 0.0,
            "by_adapter": dict(self.by_adapter),
            "successes": self.successes,
            "failures": self.failures,
            "retained_events": len(self.events),
        }
