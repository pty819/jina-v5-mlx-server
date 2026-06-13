from pathlib import Path
import unittest

from jina_v5_mlx_demo.reranking import (
    DEFAULT_RERANKER_DIR,
    OfficialMLXRerankService,
    PrismMLXReranker,
    PROJECT_DIR,
)


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


class FakePrismTokenizer:
    model_max_length = 128

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.prompts.append(messages)
        query = next(item["content"] for item in messages if item["role"] == "query")
        document = next(item["content"] for item in messages if item["role"] == "document")
        return f"query={query}\ndocument={document}"

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class ScoredPrismReranker(PrismMLXReranker):
    def __init__(self, scores):
        self.model_dir = Path(".")
        self.model = object()
        self.tokenizer = FakePrismTokenizer()
        self._scores = scores

    def _score_prompt(self, prompt: str) -> float:
        return self._scores[prompt]


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

    def test_model_id_is_prism_reranker(self):
        service = OfficialMLXRerankService(
            raw_reranker=FakeRawReranker(),
        )
        self.assertEqual(service.model_id, "pty819/prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24")

    def test_default_reranker_dir_matches_huggingface_repo_layout(self):
        self.assertEqual(
            DEFAULT_RERANKER_DIR,
            PROJECT_DIR / "models" / "pty819" / "prism-qwen3.5-reranker-0.8b-optiq-5bpw-cal24",
        )

    def test_prism_reranker_scores_each_document_and_orders_by_yes_probability(self):
        reranker = ScoredPrismReranker(
            {
                "query=boiling point\ndocument=water boils at 100 C": 0.91,
                "query=boiling point\ndocument=mountain elevation": 0.12,
                "query=boiling point\ndocument=steam temperature": 0.73,
            }
        )

        result = reranker.rerank(
            "boiling point",
            ["water boils at 100 C", "mountain elevation", "steam temperature"],
            top_n=2,
            return_embeddings=True,
        )

        self.assertEqual([item["index"] for item in result], [0, 2])
        self.assertEqual([item["relevance_score"] for item in result], [0.91, 0.73])
        self.assertEqual([item["embedding"] for item in result], [None, None])
        self.assertEqual(reranker.tokenizer.prompts[0][0]["role"], "query")
        self.assertEqual(reranker.tokenizer.prompts[0][1]["role"], "document")


if __name__ == "__main__":
    unittest.main()
