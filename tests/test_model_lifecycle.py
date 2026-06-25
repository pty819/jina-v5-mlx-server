import unittest
from unittest.mock import patch

from jina_v5_mlx_demo.modeling import DEFAULT_IDLE_SECONDS as EMBEDDING_IDLE_SECONDS
from jina_v5_mlx_demo.modeling import MLXEmbeddingService
from jina_v5_mlx_demo.reranking import DEFAULT_IDLE_SECONDS as RERANK_IDLE_SECONDS
from jina_v5_mlx_demo.reranking import MLXRerankService


class FakeEmbeddingModel:
    def encode(self, texts, tokenizer, *, task_type, truncate_dim, max_length):
        return FakeArray([[float(len(texts)), float(truncate_dim)]])


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeReranker:
    def rerank(self, query, documents, *, top_n, return_embeddings):
        return [
            {
                "index": 0,
                "document": documents[0],
                "relevance_score": 0.5,
                "embedding": None,
            }
        ]


class ModelLifecycleTest(unittest.TestCase):
    def test_embedding_and_rerank_idle_defaults_are_twenty_minutes(self):
        self.assertEqual(EMBEDDING_IDLE_SECONDS, 20 * 60)
        self.assertEqual(RERANK_IDLE_SECONDS, 20 * 60)

    def test_embedding_clears_mlx_cache_after_inference(self):
        service = MLXEmbeddingService()
        service._load = lambda: (FakeEmbeddingModel(), object())

        with (
            patch("jina_v5_mlx_demo.modeling.mx.eval"),
            patch("jina_v5_mlx_demo.modeling.mx.clear_cache") as clear_cache,
            patch("jina_v5_mlx_demo.modeling.mx.synchronize") as synchronize,
        ):
            result = service.embed(["doc"], task_type="retrieval.passage", dimensions=32)

        self.assertEqual(result, [[1.0, 32.0]])
        clear_cache.assert_called_once_with()
        synchronize.assert_called_once_with()

    def test_rerank_clears_mlx_cache_after_inference(self):
        service = MLXRerankService(
            raw_reranker=FakeReranker(),
            token_counter=lambda query, docs: 3,
        )

        with (
            patch("jina_v5_mlx_demo.reranking.mx.clear_cache") as clear_cache,
            patch("jina_v5_mlx_demo.reranking.mx.synchronize") as synchronize,
        ):
            result = service.rerank("query", ["doc"], top_n=1)

        self.assertEqual(result.results[0].relevance_score, 0.5)
        clear_cache.assert_called_once_with()
        synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
