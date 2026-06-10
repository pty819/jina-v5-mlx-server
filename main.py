import argparse
from pathlib import Path

import uvicorn

from jina_v5_mlx_demo.chat import DEFAULT_CHAT_MODEL_DIR, MLXChatService
from jina_v5_mlx_demo.chat_proxy import (
    CHAT_MODEL_ID,
    DEFAULT_CHAT_UPSTREAM_BASE_URL,
    DEFAULT_CHAT_UPSTREAM_MODEL,
    ChatProxyClient,
)
from jina_v5_mlx_demo.modeling import DEFAULT_MODEL_DIR, MLXEmbeddingService
from jina_v5_mlx_demo.reranking import DEFAULT_RERANKER_DIR, OfficialMLXRerankService
from jina_v5_mlx_demo.server import create_app


def cosine_similarity(query_embedding, passage_embeddings):
    return [sum(left * right for left, right in zip(passage, query_embedding[0])) for passage in passage_embeddings]


def run_embed(args):
    passages = args.passage or [
        "Machine learning lets computers learn patterns from data.",
        "A sourdough starter is a culture used for baking bread.",
    ]

    service = MLXEmbeddingService(args.model_dir)
    query_embedding = service.embed(
        [args.query],
        task_type="retrieval.query",
        dimensions=args.dim,
        max_length=args.max_length,
    )
    passage_embeddings = service.embed(
        passages,
        task_type="retrieval.passage",
        dimensions=args.dim,
        max_length=args.max_length,
    )

    print(f"embedding_dim={len(query_embedding[0])}")
    for score, passage in zip(cosine_similarity(query_embedding, passage_embeddings), passages):
        print(f"{score:.4f}\t{passage}")


def run_serve(args):
    embedding_service = MLXEmbeddingService(args.model_dir, idle_seconds=args.idle_seconds)
    rerank_service = OfficialMLXRerankService(args.reranker_dir, idle_seconds=args.idle_seconds)
    chat_proxy = None
    chat_service = None
    if not args.disable_chat_proxy:
        chat_proxy = ChatProxyClient(
            upstream_base_url=args.chat_upstream_base_url,
            model_id=args.chat_model,
            upstream_model=args.chat_upstream_model,
            api_key=args.chat_upstream_api_key,
            timeout_seconds=args.chat_upstream_timeout,
        )
    elif args.enable_local_chat:
        chat_service = MLXChatService(args.chat_model_dir, idle_seconds=args.idle_seconds)
    app = create_app(
        embedding_service,
        rerank_service=rerank_service,
        chat_service=chat_service,
        chat_proxy=chat_proxy,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        max_batch_tokens=args.max_batch_tokens,
        length_tolerance=args.length_tolerance,
        default_max_length=args.max_length,
        max_chat_queue_size=args.max_chat_queue_size,
        mlx_cache_limit_mb=args.mlx_cache_limit_mb,
        mlx_memory_limit_mb=args.mlx_memory_limit_mb,
        trim_mlx_cache_when_idle=not args.disable_mlx_cache_trim,
    )
    print(
        "Serving embeddings "
        f"({embedding_service.model_id}) and reranking ({rerank_service.model_id})"
    )
    if chat_proxy is not None:
        print(
            "Proxying chat "
            f"({chat_proxy.model_id}) -> {chat_proxy.upstream_base_url} "
            f"model={chat_proxy.upstream_model or chat_proxy.model_id}"
        )
    elif chat_service is not None:
        print(f"Serving local chat ({chat_service.model_id})")
    else:
        print("Chat routes disabled")
    print(f"  http://{args.host}:{args.port}")
    print("  POST /v1/embeddings, /openai/v1/embeddings, /jina/v1/embeddings")
    print("  POST /v1/rerank, /jina/v1/rerank, /openai/v1/rerank")
    if chat_proxy is not None or chat_service is not None:
        print("  POST /v1/chat/completions, /openai/v1/chat/completions")
    print("  GET  /v1/models, /openai/v1/models")
    print("  GET  /stats, /stats.json")
    uvicorn.run(app, host=args.host, port=args.port)


def run_bench_chat(args):
    messages = [
        {
            "role": "system",
            "content": "You are a professional Chinese to English translation engine. Only output English.",
        },
        {
            "role": "user",
            "content": (
                "Translate this into English while preserving meaning and tone: "
                "在真实的生产环境里，翻译模型的性能不能只看总耗时。"
                "我们需要分别观察prefill阶段处理上下文的速度，以及decode阶段逐token生成译文的速度，"
                "这样才能判断服务在长文本输入和并发请求下是否稳定。"
            ),
        },
    ]
    service = MLXChatService(args.chat_model_dir, idle_seconds=args.idle_seconds)
    try:
        result = service.complete(
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=None,
        )
    finally:
        service.close()

    print(f"model={service.model_id}")
    print(f"prompt_tokens={result['prompt_tokens']}")
    print(f"prefill_tps={result['prompt_tps']:.2f}")
    print(f"completion_tokens={result['completion_tokens']}")
    print(f"decode_tps={result['generation_tps']:.2f}")
    print(f"peak_memory_gb={result['peak_memory_gb']:.3f}")
    print("output:")
    print(result["content"].strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--reranker-dir", type=str, default=str(DEFAULT_RERANKER_DIR))
    parser.add_argument("--chat-model-dir", type=str, default=str(DEFAULT_CHAT_MODEL_DIR))
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(
        func=run_embed,
        query="What is machine learning?",
        passage=None,
        dim=1024,
        max_length=8192,
    )

    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--query", default="What is machine learning?")
    embed_parser.add_argument(
        "--passage",
        action="append",
        default=None,
    )
    embed_parser.add_argument("--dim", type=int, choices=[32, 64, 128, 256, 512, 768, 1024], default=1024)
    embed_parser.add_argument("--max-length", type=int, default=8192)
    embed_parser.set_defaults(func=run_embed)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-length", type=int, default=8192)
    serve_parser.add_argument("--max-batch-size", type=int, default=4)
    serve_parser.add_argument("--batch-timeout-ms", type=int, default=5)
    serve_parser.add_argument("--max-batch-tokens", type=int, default=8192)
    serve_parser.add_argument("--length-tolerance", type=float, default=0.2)
    serve_parser.add_argument("--idle-seconds", type=int, default=1800, help="Unload models after N seconds idle (default: 1800)")
    serve_parser.add_argument("--max-chat-queue-size", type=int, default=32, help="Maximum queued chat completion requests")
    serve_parser.add_argument("--mlx-cache-limit-mb", type=int, default=1024, help="MLX free cache limit in MB for embedding/rerank serving (default: 1024)")
    serve_parser.add_argument("--mlx-memory-limit-mb", type=int, default=None, help="Optional MLX memory limit in MB")
    serve_parser.add_argument("--disable-mlx-cache-trim", action="store_true", help="Do not clear unused MLX cache after embedding/rerank queues become idle")
    serve_parser.add_argument("--chat-model", default=CHAT_MODEL_ID, help="Public chat model id exposed by /v1/models")
    serve_parser.add_argument("--chat-upstream-base-url", default=DEFAULT_CHAT_UPSTREAM_BASE_URL, help="OpenAI-compatible vllm-mlx upstream base URL")
    serve_parser.add_argument("--chat-upstream-model", default=DEFAULT_CHAT_UPSTREAM_MODEL, help="Model name sent to the chat upstream; use empty string to preserve client model")
    serve_parser.add_argument("--chat-upstream-api-key", default=None, help="Optional bearer token for the chat upstream")
    serve_parser.add_argument("--chat-upstream-timeout", type=float, default=600.0, help="Chat upstream request timeout in seconds")
    serve_parser.add_argument("--disable-chat-proxy", action="store_true", help="Do not register chat proxy routes")
    serve_parser.add_argument("--enable-local-chat", action="store_true", help="Use the legacy in-process MLX chat service when chat proxy is disabled")
    serve_parser.set_defaults(func=run_serve)

    bench_chat_parser = subparsers.add_parser("bench-chat")
    bench_chat_parser.add_argument("--max-tokens", type=int, default=160)
    bench_chat_parser.add_argument("--temperature", type=float, default=0.0)
    bench_chat_parser.add_argument("--top-p", type=float, default=1.0)
    bench_chat_parser.add_argument("--idle-seconds", type=int, default=1800)
    bench_chat_parser.set_defaults(func=run_bench_chat)

    args = parser.parse_args()
    args.model_dir = Path(args.model_dir)
    args.reranker_dir = Path(args.reranker_dir)
    args.chat_model_dir = Path(args.chat_model_dir)
    if getattr(args, "chat_upstream_model", None) == "":
        args.chat_upstream_model = None
    args.func(args)


if __name__ == "__main__":
    main()
