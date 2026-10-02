# Base Model

Place the llama.cpp-compatible base model here as:

```text
models/base-3b-q4_k_m.gguf
```

The suggested starting point is a 3B instruct model in GGUF Q4_K_M format,
such as Qwen2.5-3B-Instruct. Obtain the weights from a source whose model
license and GGUF conversion provenance you have reviewed. The application does
not download model files.

The default `LLAMA_PROMPT_FORMAT=qwen-chatml` wraps user text in Qwen's ChatML
instruction format. Set it to `raw` for a completion model that expects raw
text instead.

The target system is a CachyOS laptop with an AMD Radeon RX 6500M and 4 GiB
VRAM. Build llama.cpp with Vulkan support and verify the GPU appears in
`llama-server --list-devices`. Begin with the configured 2048-token context and
`LLAMA_GPU_LAYERS=auto`; reduce context or choose a smaller numeric layer count
if model weights, KV cache, and adapter memory do not fit. Quantized file size
is not the same as total runtime VRAM use. Measure actual generation speed on
the target machine.

Override the filename or directory with `BASE_MODEL_FILE` or
`MODEL_BASE_PATH` in `.env`. Model weights are intentionally excluded from Git.