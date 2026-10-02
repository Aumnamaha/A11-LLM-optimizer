"""Runtime metrics for request latency and adapter usage."""

from __future__ import annotations

from typing import Any


class RequestMetrics:
    """Stores request-level latency and adapter selection data."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_request(
        self,
        prompt: str,
        adapter: str,
        latency_ms: float,
        success: bool = True,
    ) -> dict[str, Any]:
        event = {
            "prompt": prompt,
            "adapter": adapter,
            "latency_ms": float(latency_ms),
            "success": bool(success),
        }
        self.events.append(event)
        return dict(event)

    def snapshot(self) -> dict[str, Any]:
        if not self.events:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "by_adapter": {},
            }

        by_adapter: dict[str, int] = {}
        total_latency = 0.0

        for event in self.events:
            adapter = str(event["adapter"])
            by_adapter[adapter] = by_adapter.get(adapter, 0) + 1
            total_latency += float(event["latency_ms"])

        return {
            "total_requests": len(self.events),
            "avg_latency_ms": round(total_latency / len(self.events), 3),
            "by_adapter": by_adapter,
        }
