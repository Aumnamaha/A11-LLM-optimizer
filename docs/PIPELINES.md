# A11 Pipelines

## 1. Runtime Inference Pipeline

Flow: user prompt -> intent router -> select adapter -> local llama.cpp server -> response.

Run llama.cpp separately from the optimizer API. Copy `.env.example` to `.env`,
put the configured model and adapter files under `models/` and `adapters/`, then
start `./scripts/start_llama_server.sh` and `./scripts/start_optimizer.sh` in
separate terminals. The default API port is 8080 and the llama.cpp port is
8081. On the target CachyOS laptop (AMD Radeon RX 6500M, 4 GiB), build
llama.cpp with Vulkan support and verify the GPU with
`llama-server --list-devices`. `LLAMA_GPU_LAYERS=auto` is the default; tune
context and numeric GPU-layer count to fit available VRAM. The server launcher
validates that its 3B Q4 GGUF base model and all three GGUF LoRA adapters exist
before starting:

The API's `/ready` endpoint checks those files and llama.cpp's `/health` endpoint.
It returns HTTP 503 until every check succeeds. `/health` remains useful for
observing current dependency status during startup.

Adapter IDs follow the order of the `--lora` flags: Python `0`, medical `1`,
creative `2`. Adapters must be compatible with the chosen base model and the
llama.cpp build. Safetensors LoRAs need conversion to a supported GGUF format.
GPU layer selection is configurable; check performance and memory use on the
actual target hardware rather than assuming a fixed throughput.

The router picks the domain and sets scales per request. No server restart:

```json
{
  "prompt": "Write a Python script to scrape a website.",
  "n_predict": 512,
  "temperature": 0.3,
  "lora": [
    { "id": 0, "scale": 1.0 },
    { "id": 1, "scale": 0.0 },
    { "id": 2, "scale": 0.0 }
  ]
}
```

The optimizer API calls the backend at `LLAMA_SERVER_URL` (default
`http://localhost:8081`) and maps routed adapter names to these IDs. Adapters
are loaded when the backend starts; per-request switching changes LoRA scales,
not disk-to-VRAM loading. The project does not currently download or train the
base model or adapters.

`docker compose up --build` runs only the API container and connects it to the
host's llama.cpp service at `host.docker.internal:8081`. Start the model server
on the host first. The API container mounts model and adapter directories
read-only; on Linux, Docker must support the `host-gateway` host mapping.

## 2. CI/CD Pipeline

Defined in `.github/workflows/ci.yml`. Runs on pushes to all branches and on
pull requests targeting `main`:

- Blocks tracked model weights (.gguf, .safetensors, .bin, .pt, etc.) and files over 5 MB
- ruff check and ruff format --check
- mypy
- pytest (if tests/ exists)

Local equivalents live in `.githooks/` (pre-commit, pre-push). Enable once per clone:

```bash
git config core.hooksPath .githooks
```

## 3. Visual Automation Pipeline (n8n)

| Step | Node | Purpose | Config |
| --- | --- | --- | --- |
| 1. Trigger | Webhook | Receive the prompt (Telegram bot, web form) | POST, path: chat-input |
| 2. Optimizer Request | HTTP Request | Route and generate through the optimizer API | POST `http://localhost:8080/generate` with `prompt`, `max_tokens`, and `temperature` |
| 3. Output | Telegram / Email | Return the response | Map `response.content` to the message body |

