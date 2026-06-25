#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/liyifan/Library/Application Support/jina-v5-mlx-server/current"

export HOME="/Users/liyifan"
export PATH="/Users/liyifan/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONUNBUFFERED="1"
export HF_HOME="/Users/liyifan/.cache/huggingface"
export UV_CACHE_DIR="/Users/liyifan/.cache/uv"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/main.py" \
  --model-dir "${PROJECT_DIR}/models/jina-embeddings-v5-text-small-retrieval-mlx-8bit" \
  --reranker-dir "${PROJECT_DIR}/models/jinaai/jina-reranker-v3-mlx-8bit" \
  --chat-model-dir "${PROJECT_DIR}/models/Hy-MT2-1.8B-4bit" \
  serve \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch-size 4 \
  --batch-timeout-ms 5 \
  --max-batch-tokens 8192 \
  --length-tolerance 0.2 \
  --max-length 8192 \
  --idle-seconds 1200 \
  --mlx-cache-limit-mb 0 \
  --chat-upstream-base-url http://127.0.0.1:8001/v1 \
  --chat-upstream-model mlx-community/Hy-MT2-1.8B-4bit
