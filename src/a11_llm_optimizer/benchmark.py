"""Benchmarking helpers for adapter routing and request optimization."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

from .optimizer import Optimizer


class BenchmarkRunner:
    """Runs a simple prompt benchmark across known domains."""

    def __init__(
        self,
        prompts: Dict[str, Iterable[str]] | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.optimizer = optimizer or Optimizer()
        self.prompts = {
            "python": [
                "Write a Python script to scrape a website.",
                "Build a Flask API endpoint with validation.",
            ],
            "medical": [
                "Explain symptoms and treatment for diabetes.",
                "What should a patient ask about blood pressure management?",
            ],
            "creative": [
                "Write a short story about a lost astronaut.",
                "Create a poetic intro for a product launch.",
            ],
        }
        if prompts:
            self.prompts = {key: list(values) for key, values in prompts.items()}

    def run(self) -> Dict[str, List[Dict[str, object]]]:
        """Measure route selection and execution time for each prompt."""
        results: Dict[str, List[Dict[str, object]]] = {}

        for domain, prompts in self.prompts.items():
            domain_results: List[Dict[str, object]] = []
            for prompt in prompts:
                start = time.perf_counter()
                payload = self.optimizer.optimize(prompt)
                elapsed = time.perf_counter() - start
                domain_results.append(
                    {
                        "prompt": prompt,
                        "domain": payload["adapter"],
                        "adapter_path": payload["adapter_path"],
                        "latency_ms": round(elapsed * 1000, 3),
                    }
                )
            results[domain] = domain_results

        return results
