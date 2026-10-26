# Phase 0 — routing locality (`baseline/routing.json`)

The pager is worth building only if the router's expert selection is *local*
enough to cache. This is measured with `tools/model_analyzer/routing_trace.py`,
which wraps `pipeline.router.select_experts` at runtime (no engine edits) and
records every top-8 selection during prefill.

Trace: **18 tokens × 48 layers × 8 experts = 6912 selections**
(3 prompts — one repeated — 6 tokens each, all 48 layers).

## Results

| metric | value |
|---|---|
| selections | 6912 |
| distinct (layer, expert) pairs | 1916 / 6144 possible |
| distinct experts (all layers) | 128 / 128 |
| reuse factor | 3.6 selections per pair |
| **hot set covering 50 % of traffic** | **435 pairs** |
| **hot set covering 90 % of traffic** | **1381 pairs** |
| consecutive-layer Jaccard (L vs L+1, same token) | **0.033** |
| temporal Jaccard (adjacent tokens, same layer) | **0.239** |

Per (layer, expert) payload = gate 0.844 + up 0.844 + down 1.037 ≈ **2.73 MiB**.

## What the cache sizes imply

| cached pairs | payload | traffic covered |
|---|---|---|
| 435 | 1.16 GiB | 50 % |
| 1381 | **3.68 GiB** | **90 %** |
| 1916 (all observed in trace) | 5.10 GiB | 100 % of this trace |
| 6144 (whole model) | 16.35 GiB | 100 % of all experts |

**This is the encouraging result:** with the runtime's target budget (2 GB VRAM
+ 8 GB RAM ≈ 10 GiB of cache), the entire *observed* working set (5.1 GiB) fits,
and the 90 %-of-traffic hot set (3.7 GiB) fits easily. The baseline reads
~1.02 GiB of experts from SSD **per token** with ~0 % reuse; a hot-set cache
should eliminate most of that.

## Traps found (important for later phases)

1. **Cross-layer prediction is dead.** Adjacent layers share only **3.3 %** of
   experts. Phase 11's naive *"predict layer L+1 from layer L's chosen experts"*
   will not work — each layer has its own router (`ffn_gate_inp`). Prediction
   must use router **logits/weights**, frequency, or history, not the previous
   layer's *selection*.
2. **Temporal locality is only moderate (~24 %).** Adjacent tokens in a layer
   share ~1 in 4 experts, so a "repeat the previous token's experts" prefetch
   would hit ~24 % — a useful floor, not a solution. Frequency/hot-set based
   caching is stronger.
3. **Frequency is skewed but not sparse.** All 128 experts appear within 18
   tokens, so we cannot cache "the few experts"; we cache the *frequent* ones.
   The hot-set curve is the right sizing tool.

## Limitations

- Small sample: 18 tokens, prefill only, 3 prompts, one machine.
- Routing comes from the repo's own Python forward pass (real weights, verified
  parity), not llama.cpp; locality statistics should be representative but not
  byte-identical to the llama.cpp baseline.
- Generation-time locality may differ from prefill; re-trace on a long
  generation workload later (Phase 20).

## Reproduce

```bash
.venv/bin/python tools/model_analyzer/routing_trace.py \
  --model models/Qwen3-30B-A3B --tokens 6 --max-layers 48 --no-records \
  --out baseline/routing.json
```
