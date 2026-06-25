from pydantic import BaseModel, ConfigDict, model_validator


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

        if self.embedding_type != "float":
            raise ValueError("Only float embeddings are supported")
        if self.normalized is not True:
            raise ValueError("Only normalized=true is supported because the MLX model returns L2-normalized vectors")

        return self


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    documents: list[str]
    model: str | None = None
    top_n: int | None = None
    return_documents: bool = True
    return_embeddings: bool = False

    @model_validator(mode="after")
    def validate_request(self):
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.documents or not all(isinstance(item, str) for item in self.documents):
            raise ValueError("documents must be a non-empty list of strings")
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n must be a positive integer")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict] | None = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.messages:
            raise ValueError("messages must be a non-empty list")
        for message in self.messages:
            if message.role not in {"system", "user", "assistant", "tool"}:
                raise ValueError("message role must be one of system, user, assistant, tool")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if isinstance(self.stop, list) and not all(isinstance(item, str) for item in self.stop):
            raise ValueError("stop must be a string or a list of strings")
        return self


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
VALID_RERANK_MODELS = {
    "jinaai/jina-reranker-v3-mlx",
    "jina-reranker-v3-mlx",
    "jina-reranker-v3",
}
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


def ensure_rerank_model(requested: str | None, service_model_id: str):
    if requested is None:
        return
    allowed = VALID_RERANK_MODELS | {service_model_id}
    if requested not in allowed:
        raise RequestError(
            f"Unsupported rerank model '{requested}'. Supported: {', '.join(sorted(allowed))}"
        )
