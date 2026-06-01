import gc
import glob
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_embeddings.models.qwen3 import ModelArgs, Qwen3Model
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import load_config

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RERANKER_DIR = PROJECT_DIR / "models" / "jina-reranker-v3-4bit-mxfp4"
RERANK_MODEL_ID = "jina-reranker-v3-4bit-mxfp4"
DEFAULT_IDLE_SECONDS = 30 * 60
QUERY_EMBED_TOKEN_ID = 151671
DOC_EMBED_TOKEN_ID = 151670
SPECIAL_TOKENS = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token": "<|embed_token|>",
}


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


def _sanitize_input(text: str) -> str:
    for token in SPECIAL_TOKENS.values():
        text = text.replace(token, "")
    return text


def _format_docs_prompt(query: str, docs: list[str]) -> str:
    query = _sanitize_input(query)
    docs = [_sanitize_input(doc) for doc in docs]
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    doc_embed_token = SPECIAL_TOKENS["doc_embed_token"]
    query_embed_token = SPECIAL_TOKENS["query_embed_token"]
    prompt += "\n".join(
        f'<passage id="{index}">\n{doc}{doc_embed_token}\n</passage>'
        for index, doc in enumerate(docs)
    )
    prompt += f"\n<query>\n{query}{query_embed_token}\n</query>"
    return prefix + prompt + suffix


class _FourBitRankingModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Qwen3Model(args)
        self.projector = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, 512, bias=False),
        )

    def score(self, input_ids: list[int]):
        hidden_states = self.model(mx.array([input_ids]))[0]
        input_ids_np = np.array(input_ids)
        query_positions = np.where(input_ids_np == QUERY_EMBED_TOKEN_ID)[0]
        doc_positions = np.where(input_ids_np == DOC_EMBED_TOKEN_ID)[0]
        if len(query_positions) == 0:
            raise ValueError("Query embed token not found in rerank prompt")
        if len(doc_positions) == 0:
            raise ValueError("Document embed tokens not found in rerank prompt")

        query_hidden = mx.expand_dims(hidden_states[int(query_positions[0])], axis=0)
        doc_hidden = mx.stack([hidden_states[int(position)] for position in doc_positions])
        query_embeds = self.projector(query_hidden)
        doc_embeds = self.projector(doc_hidden)
        query_expanded = mx.broadcast_to(query_embeds, doc_embeds.shape)
        denominator = mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1)) * mx.sqrt(
            mx.sum(query_expanded * query_expanded, axis=-1)
        )
        scores = mx.sum(doc_embeds * query_expanded, axis=-1) / mx.maximum(denominator, 1e-9)
        return query_embeds, doc_embeds, scores


class _FourBitMLXReranker:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model, self.tokenizer = self._load_model()

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_embeddings: bool = False,
        max_doc_length: int = 2048,
        max_query_length: int = 512,
    ) -> list[dict]:
        query, truncated_docs, doc_lengths, query_length = self._truncate_texts(
            query,
            documents,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
        )
        max_length = getattr(self.tokenizer, "model_max_length", 131072)
        length_capacity = max_length - 2 * query_length
        block_size = 125

        block_docs: list[str] = []
        block_indices: list[int] = []
        doc_embeddings = []
        original_indices = []
        query_embeddings = []
        block_weights = []

        def flush_block():
            if not block_docs:
                return
            query_embeds, block_doc_embeds, block_scores = self._compute_single_batch(query, block_docs)
            mx.eval(query_embeds, block_doc_embeds, block_scores)
            scores_np = np.array(block_scores)
            query_embeddings.append(np.array(query_embeds))
            doc_embeddings.extend(np.array(block_doc_embeds))
            original_indices.extend(block_indices)
            block_weights.append(float(((1.0 + scores_np) / 2.0).max()))
            block_docs.clear()
            block_indices.clear()

        for index, (length, doc) in enumerate(zip(doc_lengths, truncated_docs)):
            block_docs.append(doc)
            block_indices.append(index)
            length_capacity -= length
            if len(block_docs) >= block_size or length_capacity <= max_doc_length:
                flush_block()
                length_capacity = max_length - 2 * query_length

        flush_block()
        query_embeddings_np = np.array(query_embeddings)
        doc_embeddings_np = np.array(doc_embeddings)
        weighted_query = np.average(query_embeddings_np, axis=0, weights=block_weights)
        scores = self._calculate_cosine_scores(weighted_query, doc_embeddings_np)
        order = np.argsort(scores)[::-1]
        limit = len(documents) if top_n is None else min(top_n, len(documents))
        return [
            {
                "document": documents[original_indices[order[i]]],
                "relevance_score": float(scores[order[i]]),
                "index": int(original_indices[order[i]]),
                "embedding": doc_embeddings_np[order[i]] if return_embeddings else None,
            }
            for i in range(limit)
        ]

    def _compute_single_batch(self, query: str, docs: list[str]):
        prompt = _format_docs_prompt(query, docs)
        input_ids = self.tokenizer.encode(prompt)
        return self.model.score(input_ids)

    def _truncate_texts(
        self,
        query: str,
        documents: list[str],
        *,
        max_query_length: int,
        max_doc_length: int,
    ):
        docs = []
        doc_lengths = []
        for doc in documents:
            doc_tokens = self.tokenizer.encode(doc)
            if len(doc_tokens) > max_doc_length:
                doc_tokens = doc_tokens[:max_doc_length]
                doc = self.tokenizer.decode(doc_tokens)
            doc_lengths.append(len(doc_tokens))
            docs.append(doc)

        query_tokens = self.tokenizer.encode(query)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]
            query = self.tokenizer.decode(query_tokens)
        return query, docs, doc_lengths, len(query_tokens)

    def _calculate_cosine_scores(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        query = np.squeeze(query_embedding, axis=0)
        return np.dot(doc_embeddings, query) / (
            np.linalg.norm(query) * np.linalg.norm(doc_embeddings, axis=1)
        )

    def _load_model(self):
        config = load_config(self.model_dir)
        args = ModelArgs.from_dict(config)
        model = _FourBitRankingModel(args)
        weights = {}
        for weight_file in sorted(glob.glob(str(self.model_dir / "model*.safetensors"))):
            loaded = mx.load(weight_file)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Expected safetensors dict from {weight_file}")
            weights.update(loaded)

        quantization = config.get("quantization")
        if quantization is not None:
            def class_predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                if hasattr(module, "weight") and module.weight.size % 64 != 0:
                    return False
                return f"{path}.scales" in weights

            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.eval()
        tokenizer = load_tokenizer(self.model_dir, {})
        return model, tokenizer


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
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Reranker not found at {model_dir}. Run: "
                    "uv run hf download mlx-community/jina-reranker-v3-4bit-mxfp4 "
                    "--local-dir models/jina-reranker-v3-4bit-mxfp4"
                )
            if not rerank_py.exists() or not projector.exists():
                raw = _FourBitMLXReranker(model_dir)
                self._raw_reranker = raw
                if hasattr(raw, "tokenizer") and self._token_counter is None:
                    tok = raw.tokenizer
                    def _count(q, docs):
                        return len(tok.encode(q)) + sum(len(tok.encode(d)) for d in docs)
                    self._token_counter = _count
                self._evictor.start()
                return raw
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
