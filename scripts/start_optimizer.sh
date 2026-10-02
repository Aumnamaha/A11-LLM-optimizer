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

if [ ! -d .venv ]; then
  python -m venv .venv
fi

. .venv/bin/activate
python -m pip install -r requirements.txt >/dev/null
PYTHONPATH=src python -m uvicorn a11_llm_optimizer.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"
