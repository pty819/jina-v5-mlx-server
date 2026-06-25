import gc
import threading
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JINA_RERANKER_REPO_ID = "jinaai/jina-reranker-v3-mlx"
DEFAULT_RERANKER_DIR = PROJECT_DIR / "models" / "jinaai" / "jina-reranker-v3-mlx"
RERANK_MODEL_ID = DEFAULT_JINA_RERANKER_REPO_ID
DEFAULT_IDLE_SECONDS = 20 * 60

# Special-token strings emitted by the Jina v3 listwise prompt template.
_QUERY_EMBED_TOKEN = "<|rerank_token|>"
_DOC_EMBED_TOKEN = "<|embed_token|>"


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


class MLPProjector(nn.Module):
    """MLP projector (1024 -> 512 -> 512) mapping backbone hidden states to embedding space.

    Mirrors the projector shipped with `jinaai/jina-reranker-v3-mlx`. Each linear
    layer may be a plain ``nn.Linear`` (fp16 source) or an ``nn.QuantizedLinear``
    (8-bit quantized source); ``_load_projector`` picks the right one based on the
    weights present in the safetensors file.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def _build_projector_linear(name: str, keys: set[str], tensors: dict) -> nn.Module:
    """Construct a (Quantized)Linear for one projector layer from saved tensors.

    ``name`` is ``linear1`` (1024->512) or ``linear2`` (512->512). If the file
    contains ``<name>.scales``/``<name>.biases`` the weights are 8-bit quantized
    (uint32-packed) and we build an ``nn.QuantizedLinear`` shell and load the
    triple into it; otherwise a plain ``nn.Linear`` for fp16 weights.
    """
    weight = tensors[f"{name}.weight"]
    scales_key = f"{name}.scales"
    biases_key = f"{name}.biases"
    if scales_key in keys and biases_key in keys:
        scales = tensors[scales_key]
        # uint32 packs 4 bytes; for 8-bit that is 4 weights per uint32.
        in_features = weight.shape[-1] * (32 // BITS_PER_UINT32_8BIT)
        out_features = weight.shape[0]
        group_size = in_features // scales.shape[-1]
        layer = nn.QuantizedLinear(
            in_features, out_features, bias=False,
            group_size=group_size, bits=8,
        )
        layer.load_weights([
            ("weight", weight),
            ("scales", scales),
            ("biases", tensors[biases_key]),
        ])
        return layer
    # Plain fp16 path.
    out_features, in_features = weight.shape
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = weight
    return layer


BITS_PER_UINT32_8BIT = 8  # number of 8-bit values packed per uint32 column


def _load_projector(projector_path: Path) -> MLPProjector:
    tensors: dict[str, mx.array] = {}
    keys: set[str] = set()
    with safe_open(str(projector_path), framework="numpy") as f:
        for k in f.keys():
            tensors[k] = mx.array(f.get_tensor(k))
            keys.add(k)
    projector = MLPProjector()
    projector.linear1 = _build_projector_linear("linear1", keys, tensors)
    projector.linear2 = _build_projector_linear("linear2", keys, tensors)
    return projector


def _format_listwise_prompt(
    query: str,
    documents: list[str],
    *,
    doc_embed_token: str,
    query_embed_token: str,
    no_thinking: bool = True,
) -> str:
    """Build the Jina v3 listwise prompt.

    All candidate passages and the query share one context window so that the
    backbone's causal self-attention lets documents attend to each other.
    """
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

    prompt = (
        f"I will provide you with {len(documents)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    doc_prompts = [
        f'<passage id="{i}">\n{doc}{doc_embed_token}\n</passage>'
        for i, doc in enumerate(documents)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{query_embed_token}\n</query>"

    return prefix + prompt + suffix


def _gather_query_hidden(hidden: mx.array, input_ids: list[int], token_id: int) -> mx.array:
    positions = np.where(np.array(input_ids) == token_id)[0]
    if positions.size == 0:
        raise ValueError("Query embed token not found in input")
    return mx.expand_dims(hidden[int(positions[0])], axis=0)


def _gather_doc_hiddens(hidden: mx.array, input_ids: list[int], token_id: int) -> mx.array:
    positions = np.where(np.array(input_ids) == token_id)[0]
    if positions.size == 0:
        raise ValueError("Document embed tokens not found in input")
    return mx.stack([hidden[int(pos)] for pos in positions])


def _cosine_scores(query_emb: mx.array, doc_embs: mx.array) -> mx.array:
    query_expanded = mx.broadcast_to(query_emb, doc_embs.shape)
    scores = mx.sum(doc_embs * query_expanded, axis=-1) / (
        mx.sqrt(mx.sum(doc_embs * doc_embs, axis=-1))
        * mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
    )
    return scores


class JinaV3Reranker:
    """Listwise reranker backed by the official `jina-reranker-v3-mlx` weights.

    Loads the Qwen3 backbone via mlx-lm (model_type=qwen3) and a separate MLP
    projector. Scoring is a single forward pass over a combined prompt, then
    cosine similarity between the projected query and document hidden states.
    """

    def __init__(self, model_dir: Path):
        from mlx_lm import load

        self.model_dir = Path(model_dir)
        self.model, self.tokenizer = load(str(self.model_dir))
        self.model.eval()
        self.projector = _load_projector(self.model_dir / "projector.safetensors")
        self._query_token_id = self.tokenizer.encode(_QUERY_EMBED_TOKEN, add_special_tokens=False)[0]
        self._doc_token_id = self.tokenizer.encode(_DOC_EMBED_TOKEN, add_special_tokens=False)[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_embeddings: bool = False,
    ) -> list[dict]:
        # return_embeddings is accepted for API compatibility but the projected
        # document vectors are never exposed; Jina v3 only needs them internally
        # to compute cosine scores.
        prompt = _format_listwise_prompt(
            query,
            documents,
            doc_embed_token=_DOC_EMBED_TOKEN,
            query_embed_token=_QUERY_EMBED_TOKEN,
            no_thinking=True,
        )
        input_ids = self.tokenizer.encode(prompt)
        # Official convention: model.model expects a list-of-lists (batch=1).
        hidden = self.model.model([input_ids])[0]

        query_emb = self.projector(_gather_query_hidden(hidden, input_ids, self._query_token_id))
        doc_embs = self.projector(_gather_doc_hiddens(hidden, input_ids, self._doc_token_id))
        scores = _cosine_scores(query_emb, doc_embs)
        mx.eval(scores)

        scored = [
            {
                "document": documents[i],
                "relevance_score": float(scores[i]),
                "index": i,
                "embedding": None,
            }
            for i in range(len(documents))
        ]
        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        limit = len(scored) if top_n is None else min(top_n, len(scored))
        return scored[:limit]


class MLXRerankService:
    """Service wrapper with lazy load, idle eviction, and MLX cache trimming.

    The public ``rerank`` signature is unchanged so the rerank queue and routes
    need no edits.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_RERANKER_DIR,
        *,
        raw_reranker=None,
        token_counter=None,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
        clear_cache_after_inference: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.model_id = RERANK_MODEL_ID
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
                    f"{DEFAULT_JINA_RERANKER_REPO_ID} "
                    "--local-dir models/jinaai/jina-reranker-v3-mlx"
                )
            raw = JinaV3Reranker(model_dir)
            self._raw_reranker = raw
            if self._token_counter is None:
                tok = raw.tokenizer

                def _count(q, docs):
                    return len(tok.encode(q)) + sum(len(tok.encode(d)) for d in docs)

                self._token_counter = _count
            self._evictor.start()
            return raw
