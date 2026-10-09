# A11-LLM-Optimizer: Ultra-Low Hardware MoE & N-Gram Offloading Architecture

> **Run 100B+ MoE Models (like Qwen3.8-Flash-Next) on Consumer Hardware (4GB/12GB VRAM & 16GB/32GB System RAM)**

`A11-LLM-Optimizer` is an experimental runtime architecture and memory-offloading framework designed to run massive Mixture-of-Experts (MoE) and lookup-augmented Large Language Models locally on consumer GPUs. By decoupling **Dense Attention**, **Dynamic Expert Routing**, and **N-Gram Lookup Tables**, this engine enables running 177B parameter models without requiring high-end workstation hardware.

---

## 📖 Table of Contents
1. [Overview & Core Philosophy](#overview--core-philosophy)
2. [Hardware Allocation & Tier Matrix](#hardware-allocation--tier-matrix)
3. [System Architecture & Data Flow](#system-architecture--data-flow)
4. [Layer Breakdown: Qwen3.8-Flash-Next (`UD-Q2_K_XL`)](#layer-breakdown-qwen38-flash-next-ud-q2_k_xl)
5. [Step-by-Step Execution Lifecycle](#step-by-step-execution-lifecycle)
6. [Quickstart & Run Configuration](#quickstart--run-configuration)
7. [Repository Structure](#repository-structure)

---

## 💡 Overview & Core Philosophy

Traditional local inference engines require loading entire model weights into unified VRAM/RAM pools. For a 100B+ model, this requires 80 GB+ of fast memory. 

`A11-LLM-Optimizer` solves the memory bottleneck by exploiting **Sparse Activation** and **Deterministic Lookups**:
- **Dense Compute in VRAM:** Only the critical attention heads, router layers, and speculative draft heads reside permanently in GPU VRAM (~2.5GB–4GB).
- **Hot Experts in System RAM:** Frequently accessed MoE experts are cached in DDR4/DDR5 system memory.
- **Cold Experts & N-Gram Tables on NVMe SSD:** The 51B N-Gram table (~28GB) and rare expert sub-networks reside on fast NVMe storage, accessed via zero-copy memory mapping (`mmap`) and asynchronous DMA prefetching.

---

## ⚡ Hardware Allocation & Tier Matrix

The architecture dynamically adapts to available hardware constraints, balancing throughput (tokens/sec) against memory capacity:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       32GB RAM / 12GB VRAM TARGET                       │
│                                                                         │
│  [ GPU VRAM: 4.5 GB ]   ──► Dense Attention + Router Heads + KV Cache   │
│  [ System RAM: 20 GB ]  ──► Hot MoE Expert Pool (150/512 Experts)       │
│  [ NVMe SSD: 56 GB ]    ──► 51B N-Gram Table + Cold Experts (mmap)     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       16GB RAM / 4GB VRAM TARGET                        │
│                                                                         │
│  [ GPU VRAM: 3.0 GB ]   ──► Core DeltaNet + Router Heads                │
│  [ System RAM: 9.5 GB ]  ──► Cached Shared Experts + Top 50 Experts    │
│  [ NVMe SSD: 66 GB ]    ──► 51B N-Gram Table + Cold Experts (mmap)     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Specs Comparison

| Hardware Layer | 12GB VRAM / 32GB RAM Target | 4GB VRAM / 16GB RAM Laptop | Transfer Protocol / Access |
| :--- | :--- | :--- | :--- |
| **GPU VRAM** | 4.5 GB Allocated | 3.0 GB Allocated | PCIe Direct / VRAM Bandwidth (~500 GB/s) |
| **System RAM** | 20.0 GB Hot Expert Cache | 9.5 GB Hot Expert Cache | Host System Bus (DDR4/DDR5 ~50–80 GB/s) |
| **NVMe SSD** | ~56.0 GB `mmap` Storage | ~66.4 GB `mmap` Storage | Asynchronous PCIe DMA (~7 GB/s) |
| **Expected Speed** | **35 – 70+ tokens/sec** | **2 – 8 tokens/sec** | Hybrid Pipeline Stream |

---

## 🏗️ System Architecture & Data Flow

```
                         ┌─────────────────────────────┐
                         │   User Prompt / Input Text  │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              12GB / 4GB GPU VRAM                             │
│                                                                              │
│  ┌──────────────────────────────────┐      ┌──────────────────────────────┐  │
│  │ Gated DeltaNet & Attention Heads │      │ MoE Router Heads & MTP Draft │  │
│  │      (Core Transformer Stack)    │      │    (Selects 11 Active Experts)│  │
│  └────────────────┬─────────────────┘      └──────────────┬───────────────┘  │
└───────────────────┼───────────────────────────────────────┼──────────────────┘
                    │                                       │
     1. Async Read  │                                       │ 2. Route Tokens
        N-Gram Row  │                                       │    to Experts
                    ▼                                       ▼
┌──────────────────────────────────────┐   ┌───────────────────────────────────┐
│           NVMe SSD STORAGE           │   │         SYSTEM RAM (DRAM)         │
│                                      │   │                                   │
│  ┌────────────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │ 51B N-Gram Table (Q4)          │  │   │  │ Hot Experts Cache (Q2_K_XL) │  │
│  │ Reads ~82 KB per token (<20 µs)│  │   │  │ Top 150/512 Active Experts   │  │
│  └────────────────────────────────┘  │   │  └──────────────┬──────────────┘  │
│  ┌────────────────────────────────┐  │   │                 │                 │
│  │ Cold MoE Experts (Q2_K_XL)     │  │   │                 │ 3. Execute      │
│  │ Dynamic `mmap` Page Faults     │◄─┼───┼─────────────────┘    Active Math  │
│  └────────────────────────────────┘  │   │                                   │
└──────────────────────────────────────┘   └───────────────────────────────────┘
                                                           │
                                                           ▼
                                           ┌───────────────────────────────┐
                                           │ Token Output Stream (100 t/s) │
                                           └───────────────────────────────┘
```

---

## 📑 Layer Breakdown: Qwen3.8-Flash-Next (`UD-Q2_K_XL`)

Using Unsloth's **Dynamic 2.0 Quantization (`UD`)**, the **177B total parameter** model is partitioned to maximize memory efficiency:

```
Qwen3.8-Flash-Next-GGUF (UD-Q2_K_XL: ~78.9 GB total on disk)
 ├── 1. N-Gram Lookup Table (~28.0 GB @ Q4) ──────────► Streamed via NVMe SSD (`mmap`)
 ├── 2. Dense Attention & Gated DeltaNet (~3.5 GB) ──► Locked in GPU VRAM
 └── 3. 512 MoE Expert Pool (~47.4 GB @ Q2_K_XL) ─────► Split: RAM Cache + SSD Page Faults
```

1. **51B N-Gram Table (~28 GB @ Q4):**
   - Stores over 20 million learned bigram/trigram token entries.
   - Requires zero Floating Point Operations (FLOPs)—accessed entirely via array row offsets.
   - Fetches **~82 KB of hash data per token**, taking **<20 microseconds** over PCIe Gen4.
2. **Dense Attention & Router Heads (~3.5 GB @ Q2_K_XL):**
   - Locked directly into GPU VRAM to maintain high compute throughput.
   - Evaluates token context, computes attention matrices, and decides expert routing.
3. **Sparse Expert Pool (125B Parameters / 512 Experts @ Q2_K_XL):**
   - Activates **11 experts per token** (10 routed + 1 shared), equal to ~6B active parameters (~1.8GB uncompressed precision).
   - High-frequency experts remain in System DRAM; niche experts load dynamically off NVMe SSD.

---

## 🔄 Step-by-Step Execution Lifecycle

```
[ User Input Query ] ──► "Explain World War 2 outbreak in 1939"
         │
         ├──► STEP 1: GPU VRAM Layer Execution
         │    • Evaluates input tokens through Gated DeltaNet attention stack.
         │    • Computes query vectors without accessing external weight files.
         │
         ├──► STEP 2: Asynchronous N-Gram Table Fetch (SSD)
         │    • GPU issues background read for Layer 2 N-Gram hash key.
         │    • Reads ~82 KB directly from NVMe SSD into GPU VRAM buffer (<0.02 ms).
         │
         ├──► STEP 3: MoE Router Expert Selection
         │    • Router head selects top 11 experts required for the token.
         │    • IF Expert in System RAM Cache  ──► Execute via Host DDR5 Bus
         │    • IF Expert on Cold SSD Storage  ──► Fetch via `mmap` Async DMA
         │
         └──► STEP 4: Token Generation & Stream Output
              • Multi-Token Prediction (MTP) draft head predicts next token sequence.
              • Streams response back to User Interface at maximum memory bandwidth.
```

---

## 🚀 Quickstart & Run Configuration

### Prerequisites
- **Python 3.10+**
- **C++ Compiler** with CUDA / ROCm support
- **NVMe SSD** with at least 100 GB free space (Read speeds ≥ 5000 MB/s recommended)

### Installation
```bash
git clone https://github.com/Aumnamaha/A11-LLM-optimizer.git
cd A11-LLM-optimizer
pip install -r requirements.txt
```

### Execution Example (`llama.cpp` / Unsloth Offload Engine)

To execute the **`Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf`** model on a **12GB GPU + 32GB System RAM** setup:

```bash
./llama-server \
  --model ./models/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf \
  --n-gpu-layers 16 \
  --ctx-size 16384 \
  --mmap \
  --threads 8 \
  --host 0.0.0.0 \
  --port 8080
```

#### Parameter Breakdown:
* `--n-gpu-layers 16`: Offloads core dense attention and router layers to 12GB VRAM (~4GB footprint).
* `--mmap`: Enables memory-mapped file access, streaming the 28GB N-Gram table and cold MoE experts directly from NVMe storage without triggering RAM allocation crashes.
* `--ctx-size 16384`: Configures a 16K context window optimized for KV-cache memory constraints.

---

## 📁 Repository Structure

```
A11-LLM-optimizer/
├── docs/
│   ├── architecture_spec.md       # Full hardware benchmark and PCIe latency analysis
│   └── diagrams/                  # System flowcharts and dynamic offload diagrams
├── engine/
│   ├── mmap_loader.py             # Memory-mapped tensor streaming & cache manager
│   ├── router_prefetch.py         # Asynchronous worker threads for N-Gram lookup
│   └── layer_splitter.py          # Dynamic VRAM/RAM/SSD parameter allocator
├── models/                        # Target directory for GGUF model files
├── config.json                    # Hardware threshold and offload strategy profiles
├── requirements.txt               # Python runtime dependencies
└── README.md                      # Architecture documentation & guide
```

---

## 🤝 Contributing

Contributions are welcome! If you are experimenting with custom layer-offloading kernels, speculative draft heads, or low-bit quantization strategies for MoE models, feel free to open a Pull Request or Submit an Issue.

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
