#!/usr/bin/env python3
"""Cross-platform pre-push verification gate for A11-LLM-Optimizer.

Runs the fast lint gate and the hermetic engine test-suite before code leaves
the machine. Works on Linux, macOS, and Windows (invoked by the pre-commit
``pre-push`` hook or manually via ``python scripts/pre_push.py``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ruff_command() -> list[str] | None:
    """Prefer a ruff executable on PATH, then the importable ``ruff`` module."""
    exe = shutil.which("ruff")
    if exe:
        return [exe]

    probe = subprocess.run(
        [sys.executable, "-c", "import ruff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "ruff"]
    return None


def _run(cmd: list[str]) -> int:
    print("==> [pre-push]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    status = 0

    ruff = _ruff_command()
    if ruff:
        status |= _run([*ruff, "check", "."])
    else:
        print("==> [pre-push] ruff not found, skipping lint "
              "(install -r requirements-dev.txt)")

    status |= _run([sys.executable, "-m", "pytest", "-q", "tests"])

    if status:
        print("==> [pre-push] checks failed", file=sys.stderr)
        return 1
    print("==> [pre-push] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
