#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

MODEL_BASE_PATH="${MODEL_BASE_PATH:-./models}"
ADAPTERS_PATH="${ADAPTERS_PATH:-./adapters}"
BASE_MODEL_FILE="${BASE_MODEL_FILE:-base-3b-q4_k_m.gguf}"
PYTHON_ADAPTER_FILE="${PYTHON_ADAPTER_FILE:-python_coder.gguf}"
MEDICAL_ADAPTER_FILE="${MEDICAL_ADAPTER_FILE:-medical_expert.gguf}"
CREATIVE_ADAPTER_FILE="${CREATIVE_ADAPTER_FILE:-creative_writer.gguf}"
MODEL_PATH="$MODEL_BASE_PATH/$BASE_MODEL_FILE"
PYTHON_ADAPTER="$ADAPTERS_PATH/$PYTHON_ADAPTER_FILE"
MEDICAL_ADAPTER="$ADAPTERS_PATH/$MEDICAL_ADAPTER_FILE"
CREATIVE_ADAPTER="$ADAPTERS_PATH/$CREATIVE_ADAPTER_FILE"

for asset in "$MODEL_PATH" "$PYTHON_ADAPTER" "$MEDICAL_ADAPTER" "$CREATIVE_ADAPTER"; do
  if [[ ! -f "$asset" ]]; then
    printf 'Required model asset not found: %s\n' "$asset" >&2
    exit 1
  fi
done

exec llama-server \
  --model "$MODEL_PATH" \
  --lora "$PYTHON_ADAPTER" \
  --lora "$MEDICAL_ADAPTER" \
  --lora "$CREATIVE_ADAPTER" \
  --lora-init-without-apply \
  --ctx-size "${LLAMA_CONTEXT_SIZE:-2048}" \
  --n-gpu-layers "${LLAMA_GPU_LAYERS:-99}" \
  --host "${LLAMA_SERVER_HOST:-0.0.0.0}" \
  --port "${LLAMA_SERVER_PORT:-8081}"