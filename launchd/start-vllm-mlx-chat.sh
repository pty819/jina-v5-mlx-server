#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/liyifan/Library/Application Support/jina-v5-mlx-server/current"

export HOME="/Users/liyifan"
export PATH="/Users/liyifan/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONUNBUFFERED="1"
export HF_HOME="/Users/liyifan/.cache/huggingface"
export UV_CACHE_DIR="/Users/liyifan/.cache/uv"

exec /Users/liyifan/.local/bin/vllm-mlx serve "${PROJECT_DIR}/models/Hy-MT2-1.8B-4bit" \
  --served-model-name mlx-community/Hy-MT2-1.8B-4bit \
  --host 127.0.0.1 \
  --port 8001 \
  --continuous-batching \
  --enable-prefix-cache \
  --use-paged-cache \
  --stream-interval 1 \
  --max-tokens 4096 \
  --max-request-tokens 4096
