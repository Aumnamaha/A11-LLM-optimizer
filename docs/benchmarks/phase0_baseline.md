# Phase 0 — baseline (llama.cpp)

Reference model: **Qwen3-30B-A3B** (30.5 B total, 3.3 B active, 128 experts,
8 active/token, 48 layers) — `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`, 17.28 GiB.

Goal of this phase (from the plan): *"What happens if we run the unmodified model?"*
— measured **before** writing any pager. These numbers fix the page size, cache
sizes, I/O method, and prefetch depth that the later phases will use.

## Method

`tools/benchmark/llama_baseline.py` drives the system `llama-bench` (build 11429,
commit d812350493) across a placement/IO matrix while sampling:

- **VRAM** — `/sys/class/drm/card1/device/mem_info_vram_used` (card1 = RX 6500).
- **RAM** — peak RSS of the process tree, plus minimum system RAM free.
- **SSD** — `nvme0n1` sectors read from `/proc/diskstats`.

A **memory guard** aborts a run (and kills the child) if free system RAM drops
below `--mem-floor-gib` (default 1.5 GiB), so a config can never trip the host
OOM fail-safe again. Records land in `baseline/benchmark.json`,
`baseline/memory.json`, `baseline/SUMMARY.md`.

Command actually used:

```bash
.venv/bin/python tools/benchmark/llama_baseline.py \
  --model models/Qwen3-30B-A3B \
  --configs configs/baseline/qwen3-30b-a3b.json \
  --prompt-tokens 32 --gen-tokens 8 --reps 1 --threads 6 \
  --timeout 700 --disk nvme0n1 --baseline-dir baseline
```

## Results

| config | status | pp tok/s | tg tok/s | peak RSS (GiB) | min free RAM (GiB) | peak VRAM RX6500 (MiB) | SSD read (GiB) | wall (s) |
|---|---|---|---|---|---|---|---|---|
| cpu_mmap | ok | 2.41 | 0.68 | 11.04 | – | 584 | 52.34 | 61.6 |
| cpu_dio | rc=-9 (OOM) | – | – | 0.14 | – | 321 | 21.54 | 52.1 |
| vulkan_attn_cpu_moe | **ok** | **3.65** | **0.91** | 11.57 | 9.92 | 987 | 44.28 | 50.1 |
| vulkan_attn_cpu_moe_dio | mem-guard | – | – | 0.11 | 0.83 | 800 | 0.30 | 10.3 |

`vulkan_attn_cpu_moe` = `-ngl 99 -mg 1 -ncmoe 999` (attention on the RX 6500,
**all** MoE experts left on CPU).

## Analysis

**The model does not fit.** 17.28 GiB of weights on ~14 GiB usable RAM means the
page cache cannot hold it, so expert tensors are almost always re-read from SSD.
Direct I/O (`-lm dio`, no page cache) cannot even start: it needs the working set
resident and is killed by the OS (`rc=-9`) or the harness guard.

**Expert traffic dominates.** From the Phase 1 analyzer:

- resident core (embed / attn / norm / router / output): **0.93 GiB (5.4 %)**
- MoE experts: **16.35 GiB (94.6 %)**
- per layer per expert: gate 0.844 MiB + up 0.844 MiB + down 1.037 MiB ≈ **2.73 MiB**

One token selects 8 experts × 48 layers = **384 expert-uses** ≈ **1.02 GiB of
expert weights per token**. For the ~40 tokens of this run that is ≈ 41 GiB — and
we measured **44.28 GiB read**, i.e. essentially **0 % cache reuse**. Token
generation is bound by NVMe reads, not compute: ~0.9 tok/s.

**The GPU helps even when experts stay on CPU.** Offloading only the resident
core (attention etc.) to the RX 6500 lifts prompt processing 2.41 → 3.65 tok/s
(+51 %) and generation 0.68 → 0.91 tok/s (+34 %) at ~987 MiB VRAM — the
2 MiB pages under-fill by ~1.0 GiB, so there is room in the 4 GiB card for a
VRAM expert cache.

## Implications for the pager

1. **Only the experts are pageable.** The resident core is < 1 GiB and should be
   pinned (VRAM preferred); this matches Phase 9 of the plan.
2. **Page size matters.** With 2 MiB pages the capacity utilization is only
   **45.4 %** (0.844 MiB and 1.037 MiB expert tensors in 2 MiB slots). A 1 MiB
   page (or expert-sized pages) would waste less — to be measured in Phase 2.
3. **The whole point is locality.** The baseline confirms ~0 % reuse, so any
   cache is pure win *if* routing is local. This is measured next
   (`baseline/routing.json`).
4. **I/O backend is not obviously mmap or dio.** `cpu_mmap` works; `dio` OOMs on
   14 GiB RAM. Phase 15 must test explicit reads and async reads, not assume.

## Reproduce

```bash
.venv/bin/python tools/benchmark/llama_baseline.py --model models/Qwen3-30B-A3B \
  --configs configs/baseline/qwen3-30b-a3b.json --prompt-tokens 32 --gen-tokens 8
# re-run just the Vulkan configs, keeping the others:
.venv/bin/python tools/benchmark/llama_baseline.py --model models/Qwen3-30B-A3B \
  --configs configs/baseline/qwen3-30b-a3b.json \
  --only vulkan_attn_cpu_moe,vulkan_attn_cpu_moe_dio
```
