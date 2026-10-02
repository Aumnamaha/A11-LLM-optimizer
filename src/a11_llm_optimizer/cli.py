"""Simple CLI for local routing and optimization."""

from __future__ import annotations

import argparse
import json

from .optimizer import Optimizer


def run_prompt(prompt: str, max_tokens: int = 256) -> dict:
    optimizer = Optimizer()
    return optimizer.optimize(prompt, max_tokens=max_tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route a prompt to the correct local adapter.")
    parser.add_argument("prompt", help="User prompt to route and optimize.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens for generation.")
    args = parser.parse_args(argv)

    result = run_prompt(args.prompt, max_tokens=args.max_tokens)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
