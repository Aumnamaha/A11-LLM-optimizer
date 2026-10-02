"""Simple quality scoring utility for routing and optimization outcomes."""

from __future__ import annotations


class QualityScorer:
    """Returns a bounded score based on routing confidence and latency."""

    def score(
        self,
        prompt: str,
        adapter: str,
        latency_ms: float,
        success: bool,
    ) -> float:
        base = 70.0
        if success:
            base += 20.0
        if adapter in {"python", "medical", "creative"}:
            base += 10.0
        if latency_ms <= 100:
            base += 5.0
        elif latency_ms <= 250:
            base += 2.0
        else:
            base -= min(15.0, latency_ms / 25.0)

        if prompt and prompt.strip():
            base += 2.0

        return max(0.0, min(100.0, round(base, 2)))
