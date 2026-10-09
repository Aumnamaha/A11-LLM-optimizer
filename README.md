# A11-LLM-Optimizer: Ultra-Low Hardware MoE & N-Gram Offloading Architecture

> **Run 100B+ MoE Models (like Qwen3.8-Flash-Next) directly from NVMe SSD on Consumer Hardware (12GB VRAM & 12GB/32GB System RAM)**

`A11-LLM-Optimizer` is a custom high-performance C++/Python LLM inference engine engineered to stream massive Mixture-of-Experts (MoE) and lookup-augmented models directly from NVMe storage. By avoiding full model weight loading into system RAM, this framework prevents OS memory thrashing/swapping and enables smooth 70+ tok/s inference on consumer hardware.

---

## 📖 Table of Contents
1. [Overview & Core Philosophy](#overview--core-philosophy)
2. [Hardware Allocation & Tier Matrix](#hardware-allocation--tier-matrix)
3. [Key System Components & Current Progress](#key-system-components--current-progress)
4. [Tool Calling & Workspace Integration](#tool-calling--workspace-integration)
5. [Layer Breakdown: Qwen3.8-Flash-Next (`UD-Q2_K_XL`)](#layer-breakdown-qwen38-flash-next-ud-q2_k_xl)
6. [Quickstart & Run Guide](#quickstart--run-guide)
7. [Repository Structure](#repository-structure)

---

## 💡 Overview & Core Philosophy

Traditional local inference runtimes attempt to map entire model weights into host system RAM or unified VRAM pools. On 70B–100B+ MoE models, this forces Linux into heavy RAM swapping and CPU lockups.

`A11-LLM-Optimizer` decouples memory requirements using a **2-Tier + Hot-Expert SSD Offloading** paradigm:
- **Permanent GPU VRAM (~4GB–12GB Budget):** Holds shared active parameters (~6B base MLPs, Attention heads, MTP heads, and MoE Router weights) alongside the active KV-Cache.
- **System RAM L2 Cache (~12GB Budget):** Holds up to ~350 high-frequency "hot" experts (coding syntax, directory inspection, tool schemas, common discourse joiners) for instant re-use.
- **NVMe SSD Storage (73GB GGUF Shards):** Acts as the primary cold store for remaining MoE experts and N-gram spec tables. Weights stream directly into GPU VRAM via zero-copy `mmap` DMA transfers and are immediately evicted post-token compute.

---

## ⚡ Hardware Allocation & Tier Matrix

The engine automatically partitions tensor allocations according to hardware budgets:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     32GB RAM / 12.8GB VRAM TARGET                       │
│                                                                         │
│  [ GPU VRAM: ~4.0 GB ]  ──► Active Base Params (~6B) + KV-Cache           │
│  [ System RAM: 12.0 GB] ──► Hot MoE Expert Pool (~350 Pinned Experts)   │
│  [ NVMe SSD: 73.0 GB ]  ──► Zero-Copy mmap Storage (Cold Experts)        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Verified Performance Benchmarks

| Hardware Target | Execution Engine | Streaming Method | Memory Footprint | Token Generation Speed |
| :--- | :--- | :--- | :--- | :--- |
| **RTX 50-Series (12.8GB VRAM) + 32GB RAM** | Native C++ (`a11_cuda_engine`) | Direct NVMe `mmap` DMA | System RAM < 4GB–12GB (No Swap) | **63.8 – 86.2 tok/s** |

---

## 🧩 Key System Components & Current Progress

- [x] **`DirectSSDMmapManager` (`engine/mmap_loader.py`):** Opens persistent file handles to sharded GGUF model files (`.gguf`), fetching targeted expert byte slices via zero-copy memory mapping without loading weights into host RAM.
- [x] **`HotExpertRAMCache` (`engine/expert_cache.py`):** Implements an LRU System RAM staging cache capped at a 12GB budget (~350 experts) to accelerate high-frequency task routing.
- [x] **`AsyncPrefetchEngine` (`engine/prefetcher.py`):** Runs a background worker thread that asynchronously pre-fetches Layer N+1 expert slices off the NVMe SSD into memory while Layer N computes.
- [x] **Native CUDA Dequantizer (`a11_cuda_engine.cpython-314-x86_64-linux-gnu.so`):** Compiled C++/CUDA C extension (`csrc/engine_cuda.cpp` and `setup.py`) for low-overhead Q2_K dequantization.
- [x] **Interactive CLI (`chat.py`):** Streaming command-line interface reporting token generation speeds, timing metrics, and cache hit/miss statistics.

---

## 🛠️ Tool Calling & Workspace Integration

The pipeline contains active local and remote tooling integrations:
1. **GitHub REST API Integration:** Automatically detects remote GitHub repository links, fetches clean file trees via REST endpoints, and parses raw source code files (e.g., `app.py`, `server.py`) using Base64 API decoding.
2. **Local Directory Scanner:** Inspects active target directories (e.g., `/home/aumnamaha/Documents/Proj/...`), extracts `README.md` context, and reads local file previews.
3. **Session Target Memory:** Persists active workspace directories and remote repository targets across multi-turn user conversation histories.

---

## 📑 Layer Breakdown: Qwen3.8-Flash-Next (`UD-Q2_K_XL`)

```
Qwen3.8-Flash-Next-GGUF (UD-Q2_K_XL: ~73.5 GB total on disk across 3 shards)
 ├── Shard 1 (0.01 GB): Metadata & Tokenizer Header Vocabulary
 ├── Shard 2 (46.55 GB): Dense Attention, Routers, and Shared Experts
 └── Shard 3 (26.90 GB): 512 MoE Expert Weights & N-Gram Tables
```

---

## 🚀 Quickstart & Run Guide

### Prerequisites
- **Linux (Fedora/Ubuntu)** with GCC/G++ and CUDA Toolkit installed
- **Python 3.10+** (Python 3.14 compatible)
- **NVMe SSD** housing sharded `.gguf` model files

### Build Native C++ CUDA Extension

Compile the custom CUDA engine binary:

```bash
python setup.py build_ext --inplace
```

### Launch Interactive Engine CLI

Run the interactive session CLI:

```bash
python chat.py
```

---

## 📁 Repository Structure

```
A11-local_language_model-optimizer/
├── csrc/
│   └── engine_cuda.cpp           # Native C++/CUDA dequantization kernel
├── engine/
│   ├── expert_cache.py           # 12GB System RAM L2 LRU Hot-Expert Cache
│   ├── layer_splitter.py         # Hardware VRAM/RAM allocation planner
│   ├── loader.py                 # Multi-shard GGUF parser & indexer
│   ├── mmap_loader.py            # Zero-copy NVMe SSD mmap DMA manager
│   ├── pipeline.py               # Main MoE execution pipeline & tool router
│   ├── prefetcher.py             # Asynchronous NVMe prefetch thread
│   ├── quant.py                  # GGUF dequantization dispatcher
│   ├── router.py                 # Top-K MoE expert selection module
│   ├── sampler.py                # Token sampling & logits evaluator
│   └── tokenizer.py              # Qwen vocabulary & token encoder
├── chat.py                       # Interactive CLI Chatbot interface
├── setup.py                      # C++/CUDA CPy extension setup script
└── README.md                     # Engine documentation & usage guide
```

---

## 🧪 Development, CI & Git Hooks

The project ships with a cross-platform toolchain (Linux, macOS, and Windows):

**CI pipeline** — `.github/workflows/ci.yml` runs on every push/PR:
- Ruff lint gate (critical syntax/undefined-name checks + advisory full report)
- Native C++ extension build, byte-compile, and engine smoke tests on an
  `ubuntu` / `macos` / `windows` × Python 3.11/3.12 matrix

**Local git hooks** — install once (any OS):

```bash
python scripts/setup_hooks.py
```

- **pre-commit:** whitespace/EOF/YAML/TOML checks, large-file guard, ruff
- **pre-push:** `scripts/pre_push.py` (ruff gate + `pytest`)

Run the checks manually at any time:

```bash
python scripts/pre_push.py        # lint + tests
pre-commit run --all-files        # all hooks
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full contribution workflow,
hook/pipeline details, and commit conventions.

---

## 📜 License
Distributed under the **MIT License**.
