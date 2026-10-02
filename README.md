
# A11-LLM-optimizer

> A local request router that sends prompts to a llama.cpp server running a shared quantized base model and preloaded LoRA adapters.

## Project Goal

This project is designed around the idea that a small shared base model stays resident, while domain-specific adapters are selected per request. The router identifies the intent of the prompt and chooses the correct adapter before generating output.

## Current Implementation Status

This repository currently includes:

- a Python package structure for the optimizer
- an intent router for prompt-to-domain matching
- an adapter registry for mapping domains to adapter files
- an optimizer orchestrator that packages request metadata
- a llama.cpp completion client with per-request adapter scales
- a pytest-based baseline to verify the routing behavior

## Repository Structure

```text
.
├── src/
│   └── a11_llm_optimizer/
│       ├── __init__.py
│       ├── api.py
│       ├── adapter_inventory.py
│       ├── adapter_registry.py
│       ├── config.py
│       ├── local_client.py
│       ├── metrics.py
│       ├── model_backend.py
│       ├── optimizer.py
│       ├── runtime_health.py
│       └── router.py
├── scripts/
│   ├── start_llama_server.sh
│   └── start_optimizer.sh
├── tests/
│   ├── test_api.py
│   ├── test_local_client.py
│   └── ...
├── adapters/README.md
├── models/README.md
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

## Usage

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

## Local Inference Setup

The target laptop has an AMD Radeon RX 6500M with 4 GiB VRAM and runs CachyOS.
Start with a 3B instruct model quantized as GGUF Q4_K_M, for example
`Qwen2.5-3B-Instruct-Q4_K_M.gguf`, and a 2048-token context. Model and adapter
weights are intentionally not included in this repository. Place a llama.cpp-
compatible base model at `models/base-3b-q4_k_m.gguf` and compatible GGUF LoRA
adapters at the paths below. Safetensors adapters must be converted to a format
supported by the selected llama.cpp build before use.
See [models/README.md](models/README.md) for the base-model target and
[adapters/README.md](adapters/README.md) for required adapter names and IDs.

Copy `.env.example` to `.env`, then place the base model and three compatible
GGUF LoRA files at the configured paths. The llama.cpp backend defaults to port
8081 and the optimizer API to port 8080. Start them in separate terminals:

```bash
./scripts/start_llama_server.sh
./scripts/start_optimizer.sh
```

Build llama.cpp with Vulkan support for the AMD GPU. From a llama.cpp source
checkout, configure and build the server with:

```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -t llama-server
```

Ensure the resulting `llama-server` is on `PATH`, and confirm its Vulkan device
is available with `llama-server --list-devices`. `LLAMA_GPU_LAYERS=auto` lets
llama.cpp select offload based on available memory; lower `LLAMA_CONTEXT_SIZE`
or set a smaller numeric `LLAMA_GPU_LAYERS` if model weights, adapter memory,
and KV cache do not fit together. The adapter IDs follow `--lora` order: Python
`0`, medical `1`, creative `2`; only the selected adapter is enabled per
request. Performance depends on the specific model, Vulkan build, and memory
pressure, so measure throughput on the laptop rather than relying on a fixed
tokens-per-second estimate.

Set `LLAMA_SERVER_URL` and `LLAMA_SERVER_TIMEOUT_SECONDS` in the environment to
override the client connection settings, then start the API with
`./scripts/start_optimizer.sh`.

The API's `/ready` endpoint returns HTTP 503 until the base model, all three
adapters, and llama.cpp's `/health` endpoint are available. `/health` reports
the current checks without requiring them to pass. For the API container,
`docker compose up --build` connects to the host-run backend through
`host.docker.internal`; start llama.cpp on the host first.

Example optimizer flow:

```python
from a11_llm_optimizer.optimizer import Optimizer

optimizer = Optimizer()
result = optimizer.optimize("Write a Python script to scrape a website.")
print(result)
```

Example output:

```python
{
    "adapter": "python",
    "adapter_path": "adapters/python_coder.gguf",
    "request": {
        "prompt": "Write a Python script to scrape a website.",
        "max_tokens": 256,
        "temperature": 0.3,
        "adapter": "python",
        "scale": 1.0,
    },
}
```

## Core Idea

1. Keep the base model loaded in memory.
2. Route the input to the best domain adapter.
3. Apply the selected adapter and generate output.
4. Keep fallback logic ready when a domain is unknown.

## Next Steps

- provide compatible, licensed base-model and domain-adapter weights
- validate adapter metadata and base-model compatibility at startup
- benchmark latency and generation quality on target hardware
- evaluate improvements beyond keyword-based intent routing

See [docs/PIPELINES.md](docs/PIPELINES.md) for the runtime and CI/CD flow.
