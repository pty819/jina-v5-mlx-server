# Jina v5 MLX Server

FastAPI serving for `jinaai/jina-embeddings-v5-text-small-retrieval-mlx` embeddings and `mlx-community/jina-reranker-v3-4bit-mxfp4` reranking on Apple Silicon with MLX, plus OpenAI-style chat proxying to a dedicated `vllm-mlx` server.

The recommended runtime is split into two processes:

- Gateway/RAG process: FastAPI embedding, rerank, stats, model discovery, and chat proxy routes.
- Chat process: `vllm-mlx serve mlx-community/Hy-MT2-1.8B-4bit` tuned for short single-turn translation requests.

## Features

- Runs Jina v5 embedding and Jina v3 4-bit reranker weights locally on macOS Apple Silicon.
- Proxies Hy-MT2 chat completions to a separate `vllm-mlx` OpenAI-compatible server.
- **Embedding endpoints:**
  - `POST /v1/embeddings`
  - `POST /openai/v1/embeddings`
  - `POST /jina/v1/embeddings`
- **Reranking endpoints:**
  - `POST /v1/rerank`
  - `POST /jina/v1/rerank`
  - `POST /openai/v1/rerank` (local Jina-shaped compatibility alias)
- **Chat completion endpoints:**
  - `POST /v1/chat/completions`
  - `POST /openai/v1/chat/completions`
- **Model discovery endpoints:**
  - `GET /v1/models`
  - `GET /openai/v1/models`
- Jina task routing for embeddings: `retrieval.query` and `retrieval.passage`.
- Matryoshka output dimensions: `32`, `64`, `128`, `256`, `512`, `768`, `1024`.
- Dynamic batching with token-length bucketing for embeddings.
- Request-level rerank queue with shared MLX inference gate.
- Chat proxy path does not share the embedding/rerank inference gate.
- Operator stats: `GET /stats` (HTML) and `GET /stats.json`.
- Optional macOS `launchd` background service.

## Requirements

- macOS on Apple Silicon.
- Python managed by `uv`.
- `vllm-mlx` installed as a separate CLI for chat serving.
- About 1.2 GB for embedding model weights.
- About 320 MB for 4-bit reranker model weights.
- About 1.1 GB for Hy-MT2 1.8B 4-bit chat model weights in the vllm-mlx process.

## Setup

```bash
uv sync

uv run hf download jinaai/jina-embeddings-v5-text-small-retrieval-mlx \
  --local-dir models/jina-embeddings-v5-text-small-retrieval-mlx

uv run hf download mlx-community/jina-reranker-v3-4bit-mxfp4 \
  --local-dir models/jina-reranker-v3-4bit-mxfp4

uv run hf download mlx-community/Hy-MT2-1.8B-4bit \
  --local-dir models/Hy-MT2-1.8B-4bit

uv tool install vllm-mlx
```

The `models/` directory is ignored by Git.

## Run

Start the chat backend first:

```bash
vllm-mlx serve models/Hy-MT2-1.8B-4bit \
  --served-model-name mlx-community/Hy-MT2-1.8B-4bit \
  --host 127.0.0.1 \
  --port 8001 \
  --continuous-batching \
  --disable-prefix-cache \
  --stream-interval 1 \
  --max-tokens 1536 \
  --max-request-tokens 1536 \
  --max-num-seqs 4 \
  --max-kv-size 3072 \
  --prefill-batch-size 4 \
  --completion-batch-size 8 \
  --prefill-step-size 1024 \
  --chunked-prefill-tokens 512 \
  --gpu-memory-utilization 0.40
```

Then start the gateway/RAG process:

```bash
uv run python main.py \
  serve \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 4 \
  --batch-timeout-ms 5 \
  --max-batch-tokens 8192 \
  --length-tolerance 0.2 \
  --max-length 8192 \
  --chat-upstream-base-url http://127.0.0.1:8001/v1 \
  --chat-upstream-model mlx-community/Hy-MT2-1.8B-4bit
```

Override model directories:

```bash
uv run python main.py \
  --model-dir /path/to/embedding-model \
  --reranker-dir /path/to/reranker-model \
  serve --host 0.0.0.0 --port 8000
```

Disable chat proxy routes entirely:

```bash
uv run python main.py serve --disable-chat-proxy
```

Use the legacy in-process chat implementation only for compatibility/debugging:

```bash
uv run python main.py --chat-model-dir models/Hy-MT2-1.8B-4bit \
  serve --disable-chat-proxy --enable-local-chat
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Embedding Endpoints

Three route aliases serve the same embedding implementation:

```text
POST /v1/embeddings
POST /openai/v1/embeddings
POST /jina/v1/embeddings
```

### OpenAI-Compatible Usage

Base URL: `http://127.0.0.1:8000/v1` or `http://127.0.0.1:8000/openai/v1`

```bash
curl http://127.0.0.1:8000/openai/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-v5-passage",
    "input": ["MLX can serve embeddings on Apple Silicon."],
    "dimensions": 256,
    "encoding_format": "float"
  }'
```

Accepted embedding model aliases:

- `jina-embeddings-v5-text-small`
- `jina-v5`, `jina-v5-passage`, `jina-v5-document`, `jina-v5-query`

Query/document routing:

- `model: "jina-v5-query"` maps to `retrieval.query`.
- `model: "jina-v5-passage"` maps to `retrieval.passage`.
- `input_type: "query"` maps to `retrieval.query`.
- `input_type: "passage"` or `"document"` maps to `retrieval.passage`.
- `task` or `task_type` can directly specify Jina tasks.

### Jina-Style Usage

```bash
curl http://127.0.0.1:8000/jina/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-embeddings-v5-text-small",
    "input": ["How do I run embeddings on a Mac?"],
    "task": "retrieval.query",
    "dimensions": 256,
    "normalized": true,
    "embedding_type": "float"
  }'
```

## Reranking Endpoints

Three route aliases serve the same Jina-shaped reranking contract:

```text
POST /v1/rerank
POST /jina/v1/rerank
POST /openai/v1/rerank
```

The `/openai/v1/rerank` route is a **local compatibility alias** that uses the same Jina-shaped request and response format. It does not implement an OpenAI-specific rerank schema.

### Rerank Request

```bash
curl http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-reranker-v3-4bit-mxfp4",
    "query": "What is MLX?",
    "documents": [
      "MLX is an array framework optimized for Apple silicon.",
      "A sourdough starter is a culture used for baking bread."
    ],
    "top_n": 1,
    "return_documents": true
  }'
```

Request fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | string | required | Non-empty query text |
| `documents` | list[string] | required | Non-empty list of documents to rank |
| `model` | string | null | Optional; accepts `jina-reranker-v3`, `jina-reranker-v3-mlx`, `jina-reranker-v3-4bit-mxfp4`, `mlx-community/jina-reranker-v3-4bit-mxfp4` |
| `top_n` | int | null | Positive integer; omit to return all results |
| `return_documents` | bool | true | Include document text in results |
| `return_embeddings` | bool | false | Include document embeddings in results |

Extra request fields are rejected.

### Rerank Response

```json
{
  "model": "jina-reranker-v3-4bit-mxfp4",
  "object": "list",
  "usage": {
    "total_tokens": 17
  },
  "results": [
    {
      "index": 0,
      "relevance_score": 0.93,
      "document": "MLX is an array framework optimized for Apple silicon."
    }
  ]
}
```

`usage.total_tokens` is counted locally using the reranker tokenizer.

The default reranker directory is `models/jina-reranker-v3-4bit-mxfp4`. The
older official `jinaai/jina-reranker-v3-mlx` directory is still supported when
passed explicitly with `--reranker-dir`, but it is no longer the default.

## Chat Completion Endpoints

Two route aliases serve the same OpenAI-style chat completion contract:

```text
POST /v1/chat/completions
POST /openai/v1/chat/completions
```

By default these routes are gateway proxy routes. They forward the OpenAI-style
request body to `--chat-upstream-base-url`, default
`http://127.0.0.1:8001/v1`, where `vllm-mlx` owns chat inference,
continuous batching, KV cache sizing, prefill/decode scheduling, and streaming.

The gateway exposes the public model id `mlx-community/Hy-MT2-1.8B-4bit` and,
by default, sends that same model id to vllm-mlx. The vllm-mlx command above
sets `--served-model-name mlx-community/Hy-MT2-1.8B-4bit` so `/v1/models`,
requests, and responses use the same visible model name. Set
`--chat-upstream-model ""` to preserve the client-supplied model unchanged.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Hy-MT2-1.8B-4bit",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Translate this into English: 今天天气真好。"}
    ],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.6
  }'
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1770000000,
  "model": "mlx-community/Hy-MT2-1.8B-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The weather is really nice today."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 9,
    "total_tokens": 41
  }
}
```

Streaming is supported with OpenAI-style server-sent events:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Hy-MT2-1.8B-4bit",
    "messages": [
      {"role": "user", "content": "Translate this into English: 我喜欢机器学习。"}
    ],
    "stream": true,
    "max_tokens": 64
  }'
```

Tool calls, vision content, and logprobs are not implemented. Multi-turn text
message arrays are accepted by the gateway; actual chat capability depends on
the vllm-mlx backend model/server.

## Model Discovery

```bash
curl http://127.0.0.1:8000/v1/models
```

The response is OpenAI-style:

```json
{
  "object": "list",
  "data": [
    {
      "id": "jina-embeddings-v5-text-small",
      "object": "model",
      "created": 0,
      "owned_by": "local",
      "capabilities": ["embeddings"]
    },
    {
      "id": "jina-reranker-v3-4bit-mxfp4",
      "object": "model",
      "created": 0,
      "owned_by": "local",
      "capabilities": ["rerank"]
    },
    {
      "id": "mlx-community/Hy-MT2-1.8B-4bit",
      "object": "model",
      "created": 0,
      "owned_by": "local",
      "capabilities": ["chat.completions"]
    }
  ]
}
```

## Chat Benchmark

Use the built-in translation benchmark to inspect prefill and decode speed:

```bash
uv run python main.py \
  --chat-model-dir models/Hy-MT2-1.8B-4bit \
  bench-chat --max-tokens 160 --temperature 0
```

The benchmark prompt is a Chinese-to-English translation request with enough
context to exercise prefill. Output includes:

- `prompt_tokens`
- `prefill_tps`
- `completion_tokens`
- `decode_tps`
- `peak_memory_gb`
- translated output text

## Stats

The service exposes operator-visible request counters and live queue state:

```text
GET /stats          (HTML page)
GET /stats.json     (JSON snapshot)
```

```bash
curl http://127.0.0.1:8000/stats.json
```

Example response:

```json
{
  "embedding": {
    "requests_1h": 12,
    "requests_1d": 31,
    "queued": 3,
    "active": 4,
    "unfinished": 7
  },
  "rerank": {
    "requests_1h": 2,
    "requests_1d": 9,
    "queued": 1,
    "active": 1,
    "unfinished": 2
  },
  "chat": {
    "requests_1h": 4,
    "requests_1d": 8,
    "queued": 2,
    "active": 1,
    "unfinished": 3
  }
}
```

- `requests_1h`: accepted requests in the last hour.
- `requests_1d`: accepted requests in the last 24 hours.
- `queued`: items waiting in queue.
- `active`: items currently being processed.
- `unfinished`: `queued + active` (operator-facing "not finished" count).

Stats are in-memory only and reset when the process restarts.

## Dynamic Batching (Embeddings)

The server runs an async batching queue for embeddings:

- Each input string becomes one queued embedding job.
- The oldest job becomes the batch anchor.
- Compatible jobs are selected when `task`, `dimensions`, and `max_length` match.
- Token length must be within `--length-tolerance` of the anchor, default `0.2`.
- Batch size is capped by `--max-batch-size`, default `4`.
- Total padded batch tokens are capped by `--max-batch-tokens`, default `8192`.
- A single input whose truncated token count exceeds `--max-batch-tokens` is rejected
  before it enters the queue.

## Process Split And Inference Gates

The gateway process keeps embedding and rerank under one process-local
`asyncio.Lock` inference gate. Only one embedding/rerank MLX model call enters
that gate at a time. This avoids local contention while preserving embedding
dynamic batching and rerank queue semantics.

Chat completions are not generated in this process by default. The gateway only
counts and forwards chat requests to vllm-mlx. vllm-mlx owns prefill/decode
scheduling, continuous batching, KV cache sizing, chunked prefill, and
streaming.

The default vllm-mlx launch script is tuned for translation-style traffic:
single-turn requests, disposable KV cache, bounded output length, and low memory
pressure. It disables prefix cache and caps request/output tokens at 1536.

The legacy in-process chat path still exists behind
`--disable-chat-proxy --enable-local-chat`, but it is not the recommended
serving path because it serializes autoregressive decode through a local FIFO
queue and shared gate.

## OpenViking Configuration

OpenViking can use this server for both dense embeddings and reranking.
`JinaDenseEmbedder` sends Jina query/passage tasks for embeddings, while the
reranker uses OpenViking's OpenAI-compatible rerank provider:

```json
{
  "embedding": {
    "dense": {
      "provider": "jina",
      "api_key": "local",
      "api_base": "http://127.0.0.1:8000/v1",
      "model": "jina-embeddings-v5-text-small",
      "dimension": 1024
    },
    "max_concurrent": 10
  },
  "rerank": {
    "provider": "openai",
    "api_key": "local",
    "api_base": "http://127.0.0.1:8000/openai/v1/rerank",
    "model": "jina-reranker-v3-4bit-mxfp4",
    "threshold": 0.1
  }
}
```

Use an address reachable from the OpenViking process. If OpenViking runs on a
different machine or container, replace `127.0.0.1` with this Mac's LAN address
or reverse-proxy hostname.

The two `api_base` fields intentionally have different shapes:

- Embedding `api_base` is a base URL such as `http://127.0.0.1:8000/v1`.
- Rerank `api_base` is the full endpoint URL. Use
  `http://127.0.0.1:8000/openai/v1/rerank` or
  `http://127.0.0.1:8000/v1/rerank`, not just an `/openai/v1` base.

OpenViking's rerank client sends `model`, `query`, and `documents`, then reads
`results[].index` and `results[].relevance_score`. This server returns that
contract from every rerank alias. Leave `top_n` unset for OpenViking so every
input document receives a score.

## macOS Background Services

The repository includes two `launchd` plists:

```text
launchd/com.liyifan.hy-mt2-vllm-mlx-chat.plist      # vllm-mlx chat backend on 127.0.0.1:8001
launchd/com.liyifan.jina-v5-mlx-embedding.plist     # gateway + embedding/rerank on 0.0.0.0:8000
```

They are intentionally separate services. The gateway does not launch
`vllm-mlx`; it only forwards `/v1/chat/completions` to
`http://127.0.0.1:8001/v1`.

The plists call these scripts:

```text
launchd/start-vllm-mlx-chat.sh
launchd/start-jina-gateway.sh
```

The scripts use absolute paths and run with `WorkingDirectory=/tmp`. They are
installed under `~/Library/Application Support/jina-v5-mlx-server/current`
because macOS LaunchAgents can hang or fail when opening files under protected
user folders such as `Documents/`.

Create/update the runtime copy:

```bash
RUNTIME_DIR="$HOME/Library/Application Support/jina-v5-mlx-server/current"
mkdir -p "$RUNTIME_DIR" ~/Library/Logs/jina-v5-mlx-server
rsync -a --delete \
  --exclude .git \
  --exclude .omx \
  --exclude .pytest_cache \
  --exclude __pycache__ \
  ./ "$RUNTIME_DIR"/
chmod +x "$RUNTIME_DIR"/launchd/start-vllm-mlx-chat.sh \
  "$RUNTIME_DIR"/launchd/start-jina-gateway.sh
```

Install the chat backend first:

```bash
CHAT_LABEL=com.liyifan.hy-mt2-vllm-mlx-chat
RUNTIME_DIR="$HOME/Library/Application Support/jina-v5-mlx-server/current"
launchctl bootout gui/$(id -u)/$CHAT_LABEL 2>/dev/null || true
cp "$RUNTIME_DIR"/launchd/$CHAT_LABEL.plist ~/Library/LaunchAgents/$CHAT_LABEL.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$CHAT_LABEL.plist
launchctl enable gui/$(id -u)/$CHAT_LABEL
launchctl kickstart -k gui/$(id -u)/$CHAT_LABEL
```

Install the gateway after the chat backend:

```bash
LABEL=com.liyifan.jina-v5-mlx-embedding
RUNTIME_DIR="$HOME/Library/Application Support/jina-v5-mlx-server/current"
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null || true
cp "$RUNTIME_DIR"/launchd/$LABEL.plist ~/Library/LaunchAgents/$LABEL.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$LABEL.plist
launchctl enable gui/$(id -u)/$LABEL
launchctl kickstart -k gui/$(id -u)/$LABEL
```

Verify:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Logs:

```text
~/Library/Logs/jina-v5-mlx-server/chat-vllm-mlx.out.log
~/Library/Logs/jina-v5-mlx-server/chat-vllm-mlx.err.log
~/Library/Logs/jina-v5-mlx-server/gateway.out.log
~/Library/Logs/jina-v5-mlx-server/gateway.err.log
```

The plists and scripts in this repository contain absolute paths for the current
machine. Update the runtime path, model paths, Python path, and `vllm-mlx` path
before using them elsewhere.

## Smoke Test

Embedding:

```bash
uv run python main.py embed --dim 256
```

Reranking (requires model weights downloaded):

```bash
uv run python -c "
from jina_v5_mlx_demo.reranking import OfficialMLXRerankService
service = OfficialMLXRerankService()
print(service.rerank('What is MLX?', ['MLX runs on Apple silicon.'], top_n=1))
"
```

## Security Notes

Binding to `0.0.0.0` exposes the service to your network. This demo does not implement authentication, TLS, rate limiting, or request body size limits. Use it on a trusted LAN or place it behind a reverse proxy/firewall if exposing it more broadly.
