import gc
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PRISM_RERANKER_REPO_ID = "pty819/prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24"
DEFAULT_RERANKER_DIR = PROJECT_DIR / "models" / "pty819" / "prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24"
PRISM_RERANK_MODEL_ID = DEFAULT_PRISM_RERANKER_REPO_ID
RERANK_MODEL_ID = PRISM_RERANK_MODEL_ID
DEFAULT_IDLE_SECONDS = 20 * 60


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    document: str | None = None
    embedding: list[float] | None = None


@dataclass
class RerankResponse:
    results: list[RerankResult]
    total_tokens: int


def _normalize_embedding(value) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return None


def _clear_mlx_cache():
    mx.clear_cache()
    mx.synchronize()


class PrismMLXReranker:
    def __init__(self, model_dir: Path, *, model=None, tokenizer=None):
        self.model_dir = Path(model_dir)
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = self._load_model()
        self._yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_embeddings: bool = False,
        max_doc_length: int = 2048,
        max_query_length: int = 512,
    ) -> list[dict]:
        query, documents = self._truncate_texts(
            query,
            documents,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
        )
        scored = []
        for index, document in enumerate(documents):
            prompt = self._build_prompt(query, document)
            scored.append(
                {
                    "document": document,
                    "relevance_score": self._score_prompt(prompt),
                    "index": index,
                    "embedding": None,
                }
            )

        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        limit = len(scored) if top_n is None else min(top_n, len(scored))
        return scored[:limit]

    def _load_model(self):
        from mlx_lm import load

        return load(str(self.model_dir))

    def _build_prompt(self, query: str, document: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def _score_prompt(self, prompt: str) -> float:
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        logits = self.model(input_ids)[:, -1, :]
        yes_no = logits[0, [self._yes_token_id, self._no_token_id]]
        probs = mx.softmax(yes_no, axis=-1)
        mx.eval(probs)
        return float(probs[0])

    def _truncate_texts(
        self,
        query: str,
        documents: list[str],
        *,
        max_query_length: int,
        max_doc_length: int,
    ):
        query_tokens = self.tokenizer.encode(query)
        if len(query_tokens) > max_query_length:
            query = self.tokenizer.decode(query_tokens[:max_query_length])

        truncated_docs = []
        for document in documents:
            doc_tokens = self.tokenizer.encode(document)
            if len(doc_tokens) > max_doc_length:
                document = self.tokenizer.decode(doc_tokens[:max_doc_length])
            truncated_docs.append(document)
        return query, truncated_docs


class OfficialMLXRerankService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_RERANKER_DIR,
        *,
        raw_reranker=None,
        token_counter=None,
        model_id: str | None = None,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
        clear_cache_after_inference: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id or _model_id_for_dir(self.model_dir)
        self._raw_reranker = raw_reranker
        self._token_counter = token_counter
        self.clear_cache_after_inference = clear_cache_after_inference
        self._load_lock = threading.Lock()
        self._evictor = IdleEvictor(
            evict=self._evict_reranker,
            idle_seconds=idle_seconds,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_embeddings: bool = False,
    ) -> RerankResponse:
        try:
            raw = self._load()
            self._evictor.touch()
            raw_results = raw.rerank(
                query,
                documents,
                top_n=top_n,
                return_embeddings=return_embeddings,
            )
            results = []
            for item in raw_results:
                results.append(RerankResult(
                    index=item["index"],
                    relevance_score=float(item["relevance_score"]),
                    document=item.get("document"),
                    embedding=_normalize_embedding(item.get("embedding")),
                ))
            total_tokens = self._count_tokens(query, documents)
            return RerankResponse(results=results, total_tokens=total_tokens)
        finally:
            if self.clear_cache_after_inference:
                _clear_mlx_cache()

    def _count_tokens(self, query: str, documents: list[str]) -> int:
        if self._token_counter is not None:
            return self._token_counter(query, documents)
        return len(query.split()) + sum(len(d.split()) for d in documents)

    def _evict_reranker(self):
        with self._load_lock:
            if self._raw_reranker is not None:
                self._raw_reranker = None
                self._token_counter = None
                gc.collect()
                _clear_mlx_cache()

    def _load(self):
        if self._raw_reranker is not None:
            return self._raw_reranker
        with self._load_lock:
            if self._raw_reranker is not None:
                return self._raw_reranker
            model_dir = self.model_dir
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Reranker not found at {model_dir}. Run: "
                    "uv run hf download "
                    f"{DEFAULT_PRISM_RERANKER_REPO_ID} "
                    "--local-dir models/pty819/prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24"
                )
            if _is_prism_reranker_dir(model_dir):
                raw = PrismMLXReranker(model_dir)
                self._raw_reranker = raw
                if hasattr(raw, "tokenizer") and self._token_counter is None:
                    tok = raw.tokenizer

                    def _count(q, docs):
                        return len(tok.encode(q)) + sum(len(tok.encode(d)) for d in docs)

                    self._token_counter = _count
                self._evictor.start()
                return raw
            raise RuntimeError(
                f"Unsupported reranker directory at {model_dir}. Expected the Prism MLX "
                f"repo layout from {DEFAULT_PRISM_RERANKER_REPO_ID}."
            )


def _is_prism_reranker_dir(model_dir: Path) -> bool:
    chat_template = model_dir / "chat_template.jinja"
    config_path = model_dir / "config.json"
    if not chat_template.exists() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
        template = chat_template.read_text()
    except OSError:
        return False
    return (
        "Qwen3_5ForCausalLM" in config.get("architectures", [])
        and '"query"' in template
        and '"document"' in template
    )


def _model_id_for_dir(model_dir: Path) -> str:
    return PRISM_RERANK_MODEL_ID
