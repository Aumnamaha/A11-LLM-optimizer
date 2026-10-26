#!/usr/bin/env python3
"""Cross-platform one-time developer setup for A11-LLM-Optimizer.

Installs the development dependencies and wires up the pre-commit + pre-push
git hooks. Works on Linux, macOS, and Windows.

Usage:
    python scripts/setup_hooks.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: list[str]) -> int:
    print("==>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    print("==> Installing development dependencies")
    status = _run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    if status != 0:
        return status

    print("==> Installing git hooks (pre-commit + pre-push)")
    status = _run([
        sys.executable, "-m", "pre_commit", "install",
        "--install-hooks",
        "--hook-type", "pre-commit",
        "--hook-type", "pre-push",
    ])
    if status != 0:
        return status

    print()
    print("Done. Hooks active:")
    print("  - pre-commit : whitespace/EOF/yaml/large-file/ruff checks")
    print("  - pre-push   : scripts/pre_push.py (ruff gate + pytest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
