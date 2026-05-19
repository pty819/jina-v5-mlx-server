from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, model_validator

from jina_v5_mlx_demo.batching import DynamicBatcher


VALID_TASK_TYPES = {
    "retrieval.query",
    "retrieval.passage",
    "classification",
    "text-matching",
    "clustering",
    "separation",
}
VALID_DIMENSIONS = {32, 64, 128, 256, 512, 768, 1024}
VALID_MAX_LENGTH_RANGE = range(1, 32769)
QUERY_MODEL_ALIASES = {
    "jina-v5-query",
    "jina-embeddings-v5-text-small-query",
}
PASSAGE_MODEL_ALIASES = {
    "jina-v5",
    "jina-v5-passage",
    "jina-v5-document",
    "jina-embeddings-v5-text-small",
    "jina-embeddings-v5-text-small-passage",
    "jina-embeddings-v5-text-small-document",
}


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str | list[str]
    model: str | None = None
    dimensions: int = 1024
    encoding_format: str = "float"
    task: str | None = None
    task_type: str | None = None
    input_type: str | None = None
    normalized: bool = True
    embedding_type: str = "float"
    max_length: int | None = None

    @model_validator(mode="after")
    def validate_request(self):
        normalize_input(self.input)
        parse_dimensions(self.dimensions)
        parse_task_type(self.task, self.task_type, self.input_type, self.model)
        if self.max_length is not None:
            parse_max_length(self.max_length)

        if self.encoding_format != "float":
            raise ValueError("Only float encoding_format is supported")
        if self.embedding_type != "float":
            raise ValueError("Only float embeddings are supported")
        if self.normalized is not True:
            raise ValueError("Only normalized=true is supported because the MLX model returns L2-normalized vectors")

        return self


def create_app(
    embedding_service,
    *,
    max_batch_size: int = 4,
    batch_timeout_ms: int = 5,
    max_batch_tokens: int = 8192,
    length_tolerance: float = 0.2,
    default_max_length: int = 8192,
) -> FastAPI:
    parse_max_length(default_max_length)
    batcher = DynamicBatcher(
        embedding_service,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        max_batch_tokens=max_batch_tokens,
        length_tolerance=length_tolerance,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await batcher.start()
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(
        title="Jina v5 MLX Embedding Server",
        version="0.1.0",
        description="Local FastAPI serving for jina-embeddings-v5-text-small-retrieval-mlx.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, error: RequestError):
        return error_response(error.status, str(error))

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, error: ValueError):
        return error_response(400, str(error))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, error: RequestValidationError):
        return error_response(400, validation_message(error))

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": embedding_service.model_id,
        }

    @app.post("/v1/embeddings")
    @app.post("/openai/v1/embeddings")
    @app.post("/jina/v1/embeddings")
    async def embeddings(request: EmbeddingRequest):
        texts = normalize_input(request.input)
        task_type = parse_task_type(request.task, request.task_type, request.input_type, request.model)
        dimensions = parse_dimensions(request.dimensions)
        max_length = parse_max_length(request.max_length or default_max_length)

        embeddings = await batcher.embed(texts, task_type=task_type, dimensions=dimensions, max_length=max_length)
        prompt_tokens = embedding_service.count_tokens(texts, task_type)

        return {
            "object": "list",
            "model": embedding_service.model_id,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding,
                }
                for index, embedding in enumerate(embeddings)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    return app


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
            }
        },
    )


def validation_message(error: RequestValidationError) -> str:
    errors = error.errors()
    if not errors:
        return "Invalid request"
    first_error = errors[0]
    context = first_error.get("ctx")
    if isinstance(context, dict) and context.get("error"):
        return str(context["error"])
    if first_error.get("loc") == ("body", "input", "str") or first_error.get("loc") == ("body", "input", "list[str]"):
        return "input must be a string or a list of strings"
    return str(first_error.get("msg", "Invalid request"))


def normalize_input(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not value:
            raise RequestError("input list must not be empty")
        return value
    raise RequestError("input must be a string or a list of strings")


def parse_dimensions(value: int) -> int:
    if not isinstance(value, int) or value not in VALID_DIMENSIONS:
        raise RequestError("dimensions must be one of 32, 64, 128, 256, 512, 768, 1024")
    return value


def parse_max_length(value: int) -> int:
    if not isinstance(value, int) or value not in VALID_MAX_LENGTH_RANGE:
        raise RequestError("max_length must be an integer between 1 and 32768")
    return value


def parse_task_type(
    task: str | None,
    task_type: str | None,
    input_type: str | None = None,
    model: str | None = None,
) -> str:
    value = task or task_type
    if value is None and input_type is not None:
        normalized_input_type = input_type.lower()
        if normalized_input_type in {"query", "retrieval.query"}:
            value = "retrieval.query"
        elif normalized_input_type in {"passage", "document", "doc", "retrieval.passage"}:
            value = "retrieval.passage"
    if value is None and model is not None:
        normalized_model = model.lower()
        if normalized_model in QUERY_MODEL_ALIASES:
            value = "retrieval.query"
        elif normalized_model in PASSAGE_MODEL_ALIASES:
            value = "retrieval.passage"
    value = value or "retrieval.passage"
    if not isinstance(value, str) or value not in VALID_TASK_TYPES:
        raise RequestError(
            "task/task_type/input_type must resolve to one of retrieval.query, retrieval.passage, "
            "classification, text-matching, clustering, separation"
        )
    if value == "separation":
        return "clustering"
    return value
