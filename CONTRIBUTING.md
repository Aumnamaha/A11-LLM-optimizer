# Contributing to A11-LLM-Optimizer

Thanks for your interest in improving **A11-LLM-Optimizer**! This guide explains
how to set up the project and how to use its three-layer automation
(**pre-commit → pre-push → CI**) so your changes land cleanly on `main`.

---

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Building the Native Extension](#building-the-native-extension)
5. [The Pipelines](#the-pipelines)
6. [Running the Checks Locally](#running-the-checks-locally)
7. [Writing Tests](#writing-tests)
8. [Code Style & Linting](#code-style--linting)
9. [Commit Conventions](#commit-conventions)
10. [Submitting Your Change](#submitting-your-change)
11. [Reporting Issues](#reporting-issues)
12. [License](#license)

---

## Ways to Contribute

- **Bug reports** – crashes, wrong output, memory/throughput regressions.
- **Features** – new loaders, dequantizers, router/sampler improvements.
- **Performance** – NVMe `mmap` streaming, cache hit-rate, tok/s benchmarks.
- **Tests** – hermetic unit tests that do not require the 70 GB GGUF shards.
- **Documentation** – README, this guide, docstrings, examples.

---

## Prerequisites

| Requirement | Notes |
| :--- | :--- |
| **Python 3.10+** | CI verifies **3.11** and **3.12**. |
| **Git** | Any recent version. |
| **C++20 compiler** | GCC / Clang / MSVC — needed to build the native extension. |
| **PyTorch** | Installed via `requirements.txt`. |
| **Node.js / NVMe / CUDA** | Only for real inference; **not** required for tests or CI. |

Development and CI run on **Linux, macOS, and Windows**. Real model execution
currently targets **Linux + NVIDIA (CUDA) + NVMe SSD**.

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/Aumnamaha/A11-LLM-optimizer.git
cd A11-LLM-optimizer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dev dependencies and wire up the git hooks (one command, any OS)
python scripts/setup_hooks.py
```

`scripts/setup_hooks.py` does two things:

1. Installs `requirements-dev.txt` (runtime deps + ruff, pytest, pre-commit, ninja).
2. Runs `pre-commit install` for the **pre-commit** and **pre-push** stages.

> Prefer a lightweight CPU dev install? Install PyTorch from the CPU index first:
> `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Building the Native Extension

The CUDA dequantizer is a C++/PyTorch extension:

```bash
python setup.py build_ext --inplace
```

- The compiler flags are chosen automatically (`setup.py` uses `/std:c++20` on
  MSVC and `-std=c++20` on GCC/Clang).
- The resulting binary is **gitignored** (`.so` / `.pyd`) — never commit build
  artifacts or `build/`.

---

## The Pipelines

This repository enforces quality through **three layers**. The first two run
locally from a single config; the third runs on every push.

### 1. Pre-commit hook — `.pre-commit-config.yaml`

Runs on **every `git commit`** against the files you staged:

| Hook | Purpose |
| :--- | :--- |
| `trailing-whitespace`, `end-of-file-fixer` | Keep diffs clean (auto-fixes). |
| `check-yaml`, `check-toml` | Validate config files. |
| `check-added-large-files` | Blocks files > 2 MB (model shards/build output). |
| `check-merge-conflict`, `mixed-line-ending` | Catch conflict markers / CRLF. |
| `detect-private-key` | Prevent committing secrets. |
| `ruff-check` | Critical Python errors (`E9,F63,F7,F82`). |

### 2. Pre-push hook — `scripts/pre_push.py`

Runs on **every `git push`** and is the last local safety net:

1. `ruff check .` — the blocking lint gate.
2. `pytest -q tests` — the hermetic engine smoke suite.

### 3. CI pipeline — `.github/workflows/ci.yml`

Runs on **push to any branch** (including newly created branches),
**pull requests**, and **manual dispatch**:

- **Lint (ruff)** — blocking critical gate plus an advisory full report.
- **Test & Build** — matrix of `ubuntu` / `macos` / `windows` × Python `3.11` /
  `3.12`; installs PyTorch, byte-compiles sources, builds the native extension,
  and runs `pytest`.

---

## Running the Checks Locally

Reproduce exactly what CI does before you push:

```bash
# Lint (same gate as CI)
ruff check --select E9,F63,F7,F82 .

# Full advisory report (shows everything, does not fail)
ruff check --statistics .

# Byte-compile
python -m compileall -q engine chat.py main.py setup.py

# Native build
python setup.py build_ext --inplace

# Tests
python -m pytest -q

# Everything at once (the pre-push gate)
python scripts/pre_push.py

# Whole pre-commit suite against every file
pre-commit run --all-files
```

> If `pre-commit run --all-files` rewrites whitespace/EOF issues, just `git add`
> the fixed files. That is the hooks doing their job.

---

## Writing Tests

- Put tests in `tests/`. Pytest config lives in `pyproject.toml`
  (`testpaths = ["tests"]`).
- **Keep them hermetic** — do not require the multi-GB GGUF shards or a GPU.
  The CI runners have neither.
- Use `tmp_path` for filesystem tests and `monkeypatch` to force the pure-Python
  / NumPy fallback paths (e.g. `engine.quant.HAS_CPP_EXTENSION = False`).
- `tests/conftest.py` already puts the repo root on `sys.path`.

Example:

```python
def test_quant_numpy_fallback(monkeypatch):
    from engine import quant
    from engine.quant import GGUFDequantizer

    monkeypatch.setattr(quant, "HAS_CPP_EXTENSION", False)
    out = GGUFDequantizer.dequantize_q2_k(bytes([0xE4]) * 4, num_elements=16)
    assert out.shape == (16,)
```

---

## Code Style & Linting

- Configuration: `pyproject.toml` → `[tool.ruff]`.
- Line length: **120**; target: **py310**.
- The **blocking** rule set is intentionally narrow (`E9,F63,F7,F82`) so the
  existing codebase stays green. Run `ruff check --select ALL` to audit more.
- Match the surrounding style: module-level docstrings, `print`-based status
  logs prefixed with the component name (e.g. `[A11-Pipeline]`).

---

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

<optional body>
```

Common types:

| Type | Use for |
| :--- | :--- |
| `feat` | New capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `docs` | Documentation only |
| `test` | Tests |
| `ci` | CI / hooks / tooling |
| `chore` | Maintenance |

Keep commits **small and focused** — one logical change per commit.

---

## Submitting Your Change

This repository develops **directly on `main`** (no mandatory feature branch):

```bash
git add <files>
git commit -m "feat: add streaming top-p sampler"
git push origin main
```

The pre-commit hook runs on commit and the pre-push gate runs on push. After the
push, **CI runs automatically** on GitHub — check the Actions tab and fix any red
result before continuing.

If you do **not** have push access:

1. Fork the repository.
2. Push your branch to your fork.
3. Open a pull request against `main` and describe the change + linked issue.

---

## Reporting Issues

Please include:

- **OS** and version, **Python** version, **PyTorch** version.
- **GPU / VRAM** and **system RAM** (for inference issues).
- The exact command you ran.
- Full traceback / logs (redact any personal paths or tokens).

---

## License

By contributing, you agree that your contributions are licensed under the
**MIT License** that covers this project.
