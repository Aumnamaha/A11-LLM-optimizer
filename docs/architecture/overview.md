# A11 Paged-MoE Runtime — Architecture Overview

## Frozen goal

> Run a large sparse MoE model on a small-VRAM GPU by keeping only the working
> expert set in VRAM, a warm subset in RAM, and the complete expert set on SSD,
> while predicting and prefetching future experts.

Explicit non-goals:

- Making SSD behave like VRAM.
- Paging the *entire* model (only MoE expert weights are aggressively pageable).

## Reference model

`Qwen3-30B-A3B` (30.5B total / 3.3B active, 128 experts, 8 active per token,
48 layers, embd 2048, expert-ffn 768). It has exactly the property this project
exploits: a large total expert pool with a small active footprint.

## Minimum supported envelope

The runtime declares a **minimum hardware requirement of 2 GB VRAM / 8 GB RAM**
(`engine.hardware.MIN_VRAM_GB`, `MIN_RAM_GB`). This is a *requirement*, not a
hard-coded budget:

- Cache tiers are **auto-sized from detected hardware** each run.
- Budgets grow with free VRAM/RAM but never overcommit currently-available
  memory, so on a bigger machine the caches are bigger.
- A machine below the envelope runs but is reported as `CONSTRAINED` with a
  warning; it is not a hard failure.

The point of the project: a **2 GB-VRAM GPU** can execute a model whose expert
pool is ~16 GiB, by caching only the working experts.

## Memory hierarchy

```
                  MODEL
                    │
        ┌───────────┴───────────┐
        │                       │
  RESIDENT CORE           MOE EXPERTS
        │                       │
      VRAM                ┌─────┴─────┐
                          │           │
                     HOT/WARM      COLD
                     CACHE         STORE
                       │             │
                      VRAM          SSD
                       ↑             ↑
                       └──── RAM ────┘
```

## Resident vs pageable

| class | tensors | placement |
|---|---|---|
| resident | embeddings, attention, norms, router (`ffn_gate_inp`), output, KV cache, runtime buffers | VRAM / pinned RAM |
| pageable | `ffn_{gate,up,down}_exps` | VRAM cache → RAM cache → SSD |

The resident set is determined by profiling (`tools/model_analyzer`), not assumed.

## RAM accounting (OOM guard)

`WeightStore` holds the resident core **dequantized to fp32** for the whole
process lifetime. For Qwen3-30B-A3B that core is 0.93 GiB quantized on disk but
**5.74 GiB in fp32** (`engine.weights.estimate_resident_fp32_bytes`). The expert
LRU cache must therefore be sized from the RAM budget **minus** that resident
footprint (`engine.weights.plan_ram_budgets`):

```
expert_cache = max(0, ram_budget - resident_fp32)
```

Sizing the expert cache from the *full* budget double-counts the core and
oversubscribes host RAM — this was the cause of the OOM (resident 5.74 GiB + a
full-budget expert cache). All forward passes also run under
`torch.inference_mode()`.

Future work: hold the resident core in a narrower dtype (fp16/bf16) or keep the
embedding/output tensors quantized, to free several GiB for the expert cache.

## Development order (do not reverse)

1. Baseline environment ← **current milestone**
2. Model analyzer
3. Page-table format
4. SSD cold-store format
5. SSD → RAM pager
6. RAM cache
7. Correctness tests (paged output == baseline output)
8. VRAM slot cache
9. RAM → VRAM transfer
10. Async I/O
11. Prefetcher
12. Adaptive cache
13. Compression
14. Telemetry
15. llama.cpp integration
16. Stress tests
17. Optimization

## Architectural rule: everything independently switchable

```
PAGING=ON/OFF   RAM_CACHE=ON/OFF   VRAM_CACHE=ON/OFF
PREFETCH=ON/OFF COMPRESSION=ON/OFF ASYNC_IO=ON/OFF
ADAPTIVE_CACHE=ON/OFF
```

Ablation ladder (Phase 19):

```
A = normal baseline
B = SSD paging only
C = SSD + RAM cache
D = SSD + RAM + VRAM cache
E = D + prefetch
F = E + async I/O
G = F + adaptive cache
H = G + compression
```

## Compute foundation

Do **not** rewrite an inference engine from scratch. Plan: use llama.cpp/ggml as
the compute foundation and build the paging/cache/prefetch layer around the
expert-loading path, until the design is proven.

## Tooling in this repo

| tool | phase | purpose |
|---|---|---|
| `tools/benchmark/llama_baseline.py` | 0 | unmodified-model baseline (pp/tg/VRAM/RSS/SSD) |
| `tools/model_analyzer/routing_trace.py` | 0 | expert-routing locality (temporal / cross-layer / hot set) |
| `tools/model_analyzer/analyze_model.py` | 1 | tensor inventory + draft page table |
| `scripts/verify_weights_parity.py` | support | WeightStore vs `gguf.quants.dequantize` |

Baseline artifacts: `baseline/{benchmark,memory,devices,routing}.json`, `baseline/SUMMARY.md`.
Analyzer artifacts: `reports/{model_inventory.json,page_table.jsonl,model_inventory.md}`.
Write-ups: `docs/benchmarks/phase0_baseline.md`, `docs/benchmarks/phase0_routing.md`.
