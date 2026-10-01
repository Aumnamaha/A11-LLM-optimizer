# A11 Pipelines

## 1. Runtime Inference Pipeline

Flow: user prompt -> intent router -> select adapter -> local llama.cpp server -> response.

Start the server once. Adapters are memory-mapped from the SSD and loaded without being applied:

```bash
llama-server -m models/base-3b.gguf \
  --lora adapters/python_coder_lora.gguf \
  --lora adapters/medical_lora.gguf \
  --lora-init-without-apply --port 8080
```

Adapter ids follow the order of the `--lora` flags (0, 1, ...). Check them with `GET /lora-adapters`.

The router picks the domain and sets scales per request. No server restart:

```json
{
  "prompt": "Write a Python script to scrape a website.",
  "n_predict": 512,
  "temperature": 0.3,
  "lora": [
    { "id": 0, "scale": 1.0 },
    { "id": 1, "scale": 0.0 }
  ]
}
```

Adapters not listed in `lora` default to scale 0.

## 2. CI/CD Pipeline

Defined in `.github/workflows/ci.yml`. Runs on push to main and on pull requests:

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
| 2. Intent Router | Switch | Pick a domain from prompt keywords | Route 1: contains "python" -> Coding. Route 2: contains "treatment" -> Medical |
| 3. Variable Setup | Set | Set the adapter id for the chosen domain | adapter_id = 0 (coding) or 1 (medical) |
| 4. Backend Request | HTTP Request | Send prompt and lora scales to the local server | POST http://localhost:8080/completion, JSON body as in section 1 |
| 5. Output | Telegram / Email | Return the response | Map the content field to the message body |

