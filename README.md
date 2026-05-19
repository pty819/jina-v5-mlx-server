# Jina v5 MLX Embedding Server

FastAPI serving for `jinaai/jina-embeddings-v5-text-small-retrieval-mlx` on Apple Silicon with MLX.

The server provides OpenAI-compatible and Jina-style embedding endpoints, supports query/document asymmetric embeddings, and batches queued requests by similar token length to avoid padding short inputs with long documents.

## Features

- Runs Jina v5 text small retrieval MLX weights locally on macOS Apple Silicon.
- OpenAI-compatible endpoint: `POST /v1/embeddings`.
- Explicit aliases: `POST /openai/v1/embeddings` and `POST /jina/v1/embeddings`.
- Jina task routing: `retrieval.query` for queries and `retrieval.passage` for documents.
- OpenAI-compatible query/document routing via `model`, `task`, `task_type`, or `input_type`.
- Matryoshka output dimensions: `32`, `64`, `128`, `256`, `512`, `768`, `1024`.
- Dynamic batching with token-length bucketing.
- Optional macOS `launchd` background service.

## Requirements

- macOS on Apple Silicon.
- Python managed by `uv`.
- About 1.2 GB for model weights.

## Setup

```bash
uv sync

uv run hf download jinaai/jina-embeddings-v5-text-small-retrieval-mlx \
  --local-dir models/jina-embeddings-v5-text-small-retrieval-mlx
```

The `models/` directory is ignored by Git.

## Run

Local-only:

```bash
uv run python main.py serve \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 4 \
  --batch-timeout-ms 5 \
  --max-batch-tokens 8192 \
  --length-tolerance 0.2 \
  --max-length 8192
```

LAN access:

```bash
uv run python main.py serve \
  --host 0.0.0.0 \
  --port 8000
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Endpoints

The same embedding implementation is exposed through three routes:

```text
POST /v1/embeddings
POST /openai/v1/embeddings
POST /jina/v1/embeddings
```

Response shape follows the OpenAI embedding response:

```json
{
  "object": "list",
  "model": "jina-embeddings-v5-text-small",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2]
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

## OpenAI-Compatible Usage

Use this base URL for OpenAI-compatible clients:

```text
http://127.0.0.1:8000/v1
```

or the explicit namespace:

```text
http://127.0.0.1:8000/openai/v1
```

Example:

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

Accepted model names and aliases:

- `jina-embeddings-v5-text-small`
- `jina-v5`
- `jina-v5-passage`
- `jina-v5-document`
- `jina-v5-query`

OpenAI's standard embedding schema has no official query/document field. This server supports several compatible conventions:

- `model: "jina-v5-query"` maps to `retrieval.query`.
- `model: "jina-v5-passage"` maps to `retrieval.passage`.
- `input_type: "query"` maps to `retrieval.query`.
- `input_type: "passage"` or `"document"` maps to `retrieval.passage`.
- `task` or `task_type` can directly specify Jina tasks.

## Jina-Style Usage

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

Supported retrieval tasks:

- `retrieval.query`
- `retrieval.passage`

The underlying MLX model also accepts `classification`, `text-matching`, and `clustering` task prefixes. The service returns normalized dense float embeddings, so `normalized: true` and `embedding_type: "float"` are the supported values.

## Dynamic Batching

The server runs an async batching queue:

- Each input string becomes one queued embedding job.
- The oldest job becomes the batch anchor.
- Compatible jobs are selected when `task`, `dimensions`, and `max_length` match.
- Token length must be within `--length-tolerance` of the anchor, default `0.2`.
- Batch size is capped by `--max-batch-size`, default `4`.
- Total padded batch tokens are capped by `--max-batch-tokens`, default `8192`.
- Results are returned to each original request in original input order.

This keeps short queries from being padded up to a long document's sequence length.

## OpenViking Configuration

Recommended dense embedding config:

```json
{
  "embedding": {
    "dense": {
      "provider": "jina",
      "api_key": "local",
      "api_base": "http://127.0.0.1:8000/v1",
      "model": "jina-embeddings-v5-text-small",
      "dimension": 1024,
      "query_param": "retrieval.query",
      "document_param": "retrieval.passage"
    },
    "max_concurrent": 10
  }
}
```

For another machine on the LAN, replace `127.0.0.1` with this Mac's IP address.

If you use `dimension: 256` or `512`, keep OpenViking's `dimension` value in sync because it is used when creating the vector schema.

## macOS Background Service

The repository includes a `launchd` plist at:

```text
launchd/com.liyifan.jina-v5-mlx-embedding.plist
```

Install it for the current user:

```bash
LABEL=com.liyifan.jina-v5-mlx-embedding
cp launchd/$LABEL.plist ~/Library/LaunchAgents/$LABEL.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$LABEL.plist
launchctl enable gui/$(id -u)/$LABEL
launchctl kickstart -k gui/$(id -u)/$LABEL
```

Check status:

```bash
launchctl print gui/$(id -u)/com.liyifan.jina-v5-mlx-embedding
```

Stop:

```bash
launchctl bootout gui/$(id -u)/com.liyifan.jina-v5-mlx-embedding
```

The plist in this repository contains absolute paths for the current machine. Update `WorkingDirectory`, log paths, and the `uv` path before using it elsewhere.

## Smoke Test

```bash
uv run python main.py embed --dim 256
```

Expected shape:

```text
embedding_dim=256
0.xxxx  Machine learning lets computers learn patterns from data.
0.xxxx  A sourdough starter is a culture used for baking bread.
```

## Security Notes

Binding to `0.0.0.0` exposes the service to your network. This demo does not implement authentication, TLS, rate limiting, or request body size limits. Use it on a trusted LAN or place it behind a reverse proxy/firewall if exposing it more broadly.
