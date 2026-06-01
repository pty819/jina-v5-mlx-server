import json
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from jina_v5_mlx_demo.chat_proxy import ChatProxyError
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


def register_chat_routes(router, chat_service, chat_queue, metrics, *, tags=None):
    _tags = tags or []

    @router.post("/chat/completions", tags=_tags, summary="Create chat completion")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            stream = await chat_queue.open_stream(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            metrics.record("chat")
            return StreamingResponse(
                _chat_completion_stream(
                    stream,
                    chat_service,
                    completion_id=completion_id,
                    created=created,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        response = await chat_queue.complete(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        metrics.record("chat")
        prompt_tokens = response["prompt_tokens"]
        completion_tokens = response["completion_tokens"]
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
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


async def _chat_completion_stream(
    stream,
    chat_service,
    *,
    completion_id,
    created,
):
    yield _sse_data({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": chat_service.model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    })
    async for chunk in stream:
        if chunk["type"] == "content" and chunk["content"]:
            yield _sse_data({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": chat_service.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk["content"]},
                        "finish_reason": None,
                    }
                ],
            })
        elif chunk["type"] == "final":
            yield _sse_data({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": chat_service.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": chunk["finish_reason"],
                    }
                ],
                "usage": {
                    "prompt_tokens": chunk["prompt_tokens"],
                    "completion_tokens": chunk["completion_tokens"],
                    "total_tokens": chunk["prompt_tokens"] + chunk["completion_tokens"],
                },
            })
    yield "data: [DONE]\n\n"


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def register_chat_proxy_routes(router, chat_proxy, metrics, *, tags=None):
    _tags = tags or []

    @router.post("/chat/completions", tags=_tags, summary="Proxy chat completion")
    async def chat_completions(request: ChatCompletionRequest):
        payload = request.model_dump(mode="json", exclude_none=True)
        if request.stream:
            try:
                stream = await chat_proxy.open_stream(payload)
            except ChatProxyError as error:
                return _chat_proxy_error_response(error)
            metrics.record("chat")
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            response = await chat_proxy.complete(payload)
        except ChatProxyError as error:
            return _chat_proxy_error_response(error)
        metrics.record("chat")
        return response


def register_model_routes(router, model_specs: list[dict], *, tags=None):
    _tags = tags or []

    @router.get("/models", tags=_tags, summary="List models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": spec["id"],
                    "object": "model",
                    "created": 0,
                    "owned_by": spec.get("owned_by", "local"),
                    "capabilities": spec.get("capabilities", []),
                }
                for spec in model_specs
            ],
        }


def _chat_proxy_error_response(error: ChatProxyError):
    payload = error.payload
    if not isinstance(payload, dict) or "error" not in payload:
        payload = {"error": {"message": str(payload), "type": "upstream_error"}}
    return JSONResponse(status_code=error.status_code, content=payload)
