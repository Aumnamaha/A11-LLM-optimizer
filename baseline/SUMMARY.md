# Baseline summary — qwen3-30b-a3b

- Model: `/home/knk/Projects/A11-LLM-optimizer/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (17.28 GiB)
- Generated: 2026-10-10T07:04:00Z
- llama.cpp: `version: 0.6.0-dev (build 11429, commit d812350493)`
- Configs: 32 prompt tok, 8 gen tok, 1 reps, 6 threads

## Results

| config | status | pp tok/s | tg tok/s | peak RSS (GiB) | min free RAM (GiB) | peak VRAM RX6500 (MiB) | SSD read (GiB) | wall (s) |
|---|---|---|---|---|---|---|---|---|
| cpu_mmap | ok | 2.41 | 0.68 | 11.04 | - | 584 | 52.34 | 61.6 |
| cpu_dio | rc=-9 | - | - | 0.14 | - | 321 | 21.54 | 52.1 |
| vulkan_attn_cpu_moe | ok | 3.65 | 0.91 | 11.57 | 9.92 | 987 | 44.28 | 50.1 |
| vulkan_attn_cpu_moe_dio | mem-guard | - | - | 0.11 | 0.83 | 800 | 0.30 | 10.3 |

## Notes

- VRAM RX6500 = `/sys/class/drm/card1/device/mem_info_vram_used` peak.
- `pp` = prompt processing throughput; `tg` = token generation throughput.
- `SSD read` is NVMe sectors read during the run (page-in from the GGUF).
- `mem-guard` = run aborted by the harness RAM floor before the host OOM fail-safe fired.
- A raw `rc=-9` means the host OOM killer terminated the process (RAM insufficient).
- This table is the *reference* for the paged-MoE runtime ablations (Phase 19).
