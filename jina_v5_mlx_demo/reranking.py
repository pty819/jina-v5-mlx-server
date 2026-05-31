import gc
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RERANKER_DIR = PROJECT_DIR / "models" / "jina-reranker-v3-mlx"
RERANK_MODEL_ID = "jina-reranker-v3"
DEFAULT_IDLE_SECONDS = 30 * 60


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


class OfficialMLXRerankService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_RERANKER_DIR,
        *,
        raw_reranker=None,
        token_counter=None,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
    ):
        self.model_id = RERANK_MODEL_ID
        self.model_dir = Path(model_dir)
        self._raw_reranker = raw_reranker
        self._token_counter = token_counter
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
                mx.clear_cache()
                mx.synchronize()

    def _load(self):
        if self._raw_reranker is not None:
            return self._raw_reranker
        with self._load_lock:
            if self._raw_reranker is not None:
                return self._raw_reranker
            model_dir = self.model_dir
            rerank_py = model_dir / "rerank.py"
            projector = model_dir / "projector.safetensors"
            if not model_dir.exists() or not rerank_py.exists() or not projector.exists():
                raise FileNotFoundError(
                    f"Official reranker not found at {model_dir}. Run: "
                    "uv run hf download jinaai/jina-reranker-v3-mlx "
                    "--local-dir models/jina-reranker-v3-mlx"
                )
            spec = importlib.util.spec_from_file_location(
                "jina_mlx_reranker",
                rerank_py,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load reranker implementation from {rerank_py}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            raw = module.MLXReranker(
                model_path=str(model_dir),
                projector_path=str(projector),
            )
            self._raw_reranker = raw
            if hasattr(raw, "tokenizer") and self._token_counter is None:
                tok = raw.tokenizer
                def _count(q, docs):
                    return len(tok.encode(q)) + sum(len(tok.encode(d)) for d in docs)
                self._token_counter = _count
            self._evictor.start()
            return raw
