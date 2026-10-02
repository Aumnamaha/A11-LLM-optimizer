"""Intent routing and request shaping for adapter-based model inference."""

from __future__ import annotations

from typing import Dict, List


class IntentRouter:
    """Simple keyword-based router that maps prompts to a target adapter."""

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "python": [
            "python",
            "script",
            "code",
            "api",
            "function",
            "debug",
            "loop",
            "class",
            "import",
            "pip",
        ],
        "medical": [
            "medical",
            "doctor",
            "symptom",
            "diagnosis",
            "treatment",
            "diabetes",
            "patient",
            "health",
            "clinic",
        ],
        "creative": [
            "story",
            "poem",
            "write",
            "creative",
            "fiction",
            "blog",
            "novel",
            "ad copy",
        ],
    }

    def route(self, prompt: str) -> str:
        """Return the best adapter domain for the prompt."""
        if not prompt:
            return "general"

        normalized = prompt.lower()

        best_domain = "general"
        best_score = 0

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in normalized)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def build_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        adapter: str | None = None,
        scale: float = 1.0,
    ) -> Dict[str, object]:
        """Build a structured request payload for the local server."""
        selected_adapter = adapter or self.route(prompt)

        return {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "adapter": selected_adapter,
            "scale": float(scale),
        }
