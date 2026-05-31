import time
import uuid

from jina_v5_mlx_demo.schema import (
    ChatCompletionRequest,
    EmbeddingRequest,
    RerankRequest,
    ensure_rerank_model,
    normalize_input,
    parse_dimensions,
    parse_max_length,
    parse_task_type,
)


def register_embedding_routes(router, embedding_service, batcher, metrics, default_max_length, *, tags=None):
    _tags = tags or []

    @router.post("/embeddings", tags=_tags, summary="Create embeddings")
    async def embeddings(request: EmbeddingRequest):
        texts = normalize_input(request.input)
        task_type = parse_task_type(request.task, request.task_type, request.input_type, request.model)
        dimensions = parse_dimensions(request.dimensions)
        max_length = parse_max_length(request.max_length or default_max_length)

        metrics.record("embedding")
        embeddings_result = await batcher.embed(texts, task_type=task_type, dimensions=dimensions, max_length=max_length)
        prompt_tokens = embedding_service.count_tokens(texts, task_type)

        return {
            "object": "list",
            "model": embedding_service.model_id,
            "data": [
                {"object": "embedding", "index": i, "embedding": e}
                for i, e in enumerate(embeddings_result)
            ],
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }


def register_rerank_routes(router, rerank_service, rerank_queue, metrics, *, tags=None):
    _tags = tags or []

    @router.post("/rerank", tags=_tags, summary="Rerank documents")
    async def rerank(request: RerankRequest):
        ensure_rerank_model(request.model, rerank_service.model_id)
        metrics.record("rerank")
        response = await rerank_queue.rerank(
            request.query,
            request.documents,
            top_n=request.top_n,
            return_embeddings=request.return_embeddings,
        )
        return _rerank_response_payload(
            model=rerank_service.model_id,
            response=response,
            return_documents=request.return_documents,
            return_embeddings=request.return_embeddings,
        )


def _rerank_response_payload(*, model, response, return_documents, return_embeddings):
    results = []
    for r in response.results:
        item = {"index": r.index, "relevance_score": r.relevance_score}
        if return_documents:
            item["document"] = r.document
        if return_embeddings:
            item["embedding"] = r.embedding
        results.append(item)
    return {
        "model": model,
        "object": "list",
        "usage": {"total_tokens": response.total_tokens},
        "results": results,
    }


def register_chat_routes(router, chat_service, chat_queue, *, tags=None):
    _tags = tags or []

    @router.post("/chat/completions", tags=_tags, summary="Create chat completion")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        response = await chat_queue.complete(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        prompt_tokens = response["prompt_tokens"]
        completion_tokens = response["completion_tokens"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_service.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["content"],
                    },
                    "finish_reason": response["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
