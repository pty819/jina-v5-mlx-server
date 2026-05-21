import unittest

from jina_v5_mlx_demo.reranking import RerankResult, RerankResponse, OfficialMLXRerankService


class FakeRawReranker:
    def rerank(self, query, documents, top_n=None, return_embeddings=False):
        return [
            {
                "index": 1,
                "relevance_score": 0.9,
                "document": documents[1],
                "embedding": [0.1, 0.2] if return_embeddings else None,
            }
        ]


class ArrayLike:
    def tolist(self):
        return [1, 2.5]


class ArrayLikeRawReranker:
    def rerank(self, query, documents, top_n=None, return_embeddings=False):
        return [
            {
                "index": 0,
                "relevance_score": 0.7,
                "document": documents[0],
                "embedding": ArrayLike() if return_embeddings else None,
            }
        ]


class RerankingServiceTest(unittest.TestCase):
    def test_normalizes_rerank_results(self):
        service = OfficialMLXRerankService(
            raw_reranker=FakeRawReranker(),
            token_counter=lambda query, docs: 7,
        )

        result = service.rerank(
            "question",
            ["first", "second"],
            top_n=1,
            return_embeddings=True,
        )

        self.assertEqual(result.total_tokens, 7)
        self.assertEqual(result.results[0].index, 1)
        self.assertEqual(result.results[0].embedding, [0.1, 0.2])

    def test_converts_array_like_embeddings_to_float_lists(self):
        service = OfficialMLXRerankService(
            raw_reranker=ArrayLikeRawReranker(),
            token_counter=lambda query, docs: 3,
        )

        result = service.rerank(
            "q",
            ["doc"],
            top_n=1,
            return_embeddings=True,
        )

        self.assertEqual(result.results[0].embedding, [1.0, 2.5])

    def test_omits_embedding_when_not_requested(self):
        service = OfficialMLXRerankService(
            raw_reranker=FakeRawReranker(),
            token_counter=lambda query, docs: 5,
        )

        result = service.rerank(
            "q",
            ["a", "b"],
            top_n=1,
            return_embeddings=False,
        )

        self.assertIsNone(result.results[0].embedding)

    def test_default_token_counter_counts_words(self):
        service = OfficialMLXRerankService(
            raw_reranker=FakeRawReranker(),
        )

        result = service.rerank(
            "What is MLX?",
            ["MLX is an array framework.", "Another doc."],
            top_n=1,
        )

        # "What is MLX?" = 3 words + "MLX is an array framework." = 5 words + "Another doc." = 2 words = 10
        self.assertEqual(result.total_tokens, 10)

    def test_model_id_is_jina_reranker_v3(self):
        service = OfficialMLXRerankService(
            raw_reranker=FakeRawReranker(),
        )
        self.assertEqual(service.model_id, "jina-reranker-v3")


if __name__ == "__main__":
    unittest.main()
