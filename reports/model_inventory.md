# Model inventory — `Qwen3-30B-A3B`

- Shards: Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
- Layers: 48 · Experts: 128 · Active/token: 8

## Size split

| category | bytes | GiB | share |
|---|---|---|---|
| resident core | 997554176 | 0.93 | 5.38% |
| MoE experts | 17553162240 | 16.35 | 94.62% |
| **total** | 18550716416 | 17.28 | 100% |

## Page model

- Page size: 2 MiB
- Total logical pages: **18432** (1 per expert tensor)
- Page-table estimate: 1179648 bytes (~1152 KiB)
- Page capacity utilization: 45.4%

## Expert bytes per kind

| kind | total (all layers) | per layer, per expert |
|---|---|---|
| down | 6372.00 MiB | 1.037 MiB |
| gate | 5184.00 MiB | 0.844 MiB |
| up | 5184.00 MiB | 0.844 MiB |

## Largest resident tensors

| tensor | type | shape | MiB |
|---|---|---|---|
| `output.weight` | 14 | [2048, 151936] | 243.4 |
| `token_embd.weight` | 12 | [2048, 151936] | 166.9 |
| `blk.0.attn_output.weight` | 12 | [4096, 2048] | 4.5 |
| `blk.0.attn_q.weight` | 12 | [2048, 4096] | 4.5 |
| `blk.1.attn_output.weight` | 12 | [4096, 2048] | 4.5 |
| `blk.1.attn_q.weight` | 12 | [2048, 4096] | 4.5 |
| `blk.2.attn_output.weight` | 12 | [4096, 2048] | 4.5 |
| `blk.2.attn_q.weight` | 12 | [2048, 4096] | 4.5 |
| `blk.3.attn_output.weight` | 12 | [4096, 2048] | 4.5 |
| `blk.3.attn_q.weight` | 12 | [2048, 4096] | 4.5 |
| `blk.4.attn_output.weight` | 12 | [4096, 2048] | 4.5 |
| `blk.4.attn_q.weight` | 12 | [2048, 4096] | 4.5 |
