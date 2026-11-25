#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
RUNTIME_ROOT="/Users/akash/Documents/Documents - AKASH's Macbook Pro/Codex/2026-11-09/https-claude-ai-share-5055aef7-9597/work"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME_ROOT/codet5-smoke-deps${PYTHONPATH:+:$PYTHONPATH}"
exec "$RUNTIME_ROOT/dataset-env/bin/python" -B "$PROJECT_ROOT/evaluate.py" "$@"
