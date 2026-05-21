import argparse
from pathlib import Path

import uvicorn

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
    app = create_app(
        embedding_service,
        rerank_service=rerank_service,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        max_batch_tokens=args.max_batch_tokens,
        length_tolerance=args.length_tolerance,
        default_max_length=args.max_length,
    )
    print(f"Serving embeddings ({embedding_service.model_id}) and reranking ({rerank_service.model_id})")
    print(f"  http://{args.host}:{args.port}")
    print("  POST /v1/embeddings, /openai/v1/embeddings, /jina/v1/embeddings")
    print("  POST /v1/rerank, /jina/v1/rerank, /openai/v1/rerank")
    print("  GET  /stats, /stats.json")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--reranker-dir", type=str, default=str(DEFAULT_RERANKER_DIR))
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
    serve_parser.set_defaults(func=run_serve)

    args = parser.parse_args()
    args.model_dir = Path(args.model_dir)
    args.reranker_dir = Path(args.reranker_dir)
    args.func(args)


if __name__ == "__main__":
    main()
