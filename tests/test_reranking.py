import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from jina_v5_mlx_demo.reranking import (
    DEFAULT_JINA_RERANKER_REPO_ID,
    DEFAULT_RERANKER_DIR,
    JinaV3Reranker,
    MLXRerankService,
    MLPProjector,
    PROJECT_DIR,
    RERANK_MODEL_ID,
    _cosine_scores,
    _format_listwise_prompt,
    _gather_doc_hiddens,
    _gather_query_hidden,
)
import mlx.core as mx


class FakeRawReranker:
    """Stand-in for JinaV3Reranker; returns one fixed ranked result."""

    def __init__(self, *, index=1, score=0.9, embedding=None):
        self._index = index
        self._score = score
        self._embedding = embedding

    def rerank(self, query, documents, *, top_n=None, return_embeddings=False):
        return [
            {
                "index": self._index,
                "relevance_score": self._score,
                "document": documents[self._index],
                "embedding": self._embedding,
            }
        ]


class FormatPromptTest(unittest.TestCase):
    def test_prompt_contains_each_passage_and_query_with_special_tokens(self):
        prompt = _format_listwise_prompt(
            "what is mlx",
            ["doc one", "doc two"],
            doc_embed_token="<|embed_token|>",
            query_embed_token="<|rerank_token|>",
        )
        self.assertIn('<passage id="0">', prompt)
        self.assertIn("doc one<|embed_token|>", prompt)
        self.assertIn('<passage id="1">', prompt)
        self.assertIn("doc two<|embed_token|>", prompt)
        self.assertIn("<query>", prompt)
        self.assertIn("what is mlx<|rerank_token|>", prompt)
        # no_thinking default collapses the think block
        self.assertIn("<think>\n\n</think>", prompt)

    def test_no_thinking_false_omits_collapsed_think_block(self):
        prompt = _format_listwise_prompt(
            "q", ["d"], doc_embed_token="<d>", query_embed_token="<q>", no_thinking=False
        )
        self.assertNotIn("<think>\n\n</think>", prompt)

    def test_passage_count_in_header_matches_documents(self):
        prompt = _format_listwise_prompt(
            "q", ["a", "b", "c"],
            doc_embed_token="<d>", query_embed_token="<q>",
        )
        self.assertIn("I will provide you with 3 passages", prompt)


class GatherHiddenTest(unittest.TestCase):
    def setUp(self):
        # 4 tokens, hidden_size 2
        self.hidden = mx.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.input_ids = [10, 11, 10, 12]

    def test_gathers_first_query_token_position(self):
        result = _gather_query_hidden(self.hidden, self.input_ids, token_id=11)
        self.assertEqual(result.shape, (1, 2))
        mx.eval(result)
        self.assertEqual(result.tolist(), [[1.0, 1.0]])

    def test_gathers_all_doc_token_positions_in_order(self):
        result = _gather_doc_hiddens(self.hidden, self.input_ids, token_id=10)
        self.assertEqual(result.shape, (2, 2))
        mx.eval(result)
        self.assertEqual(result.tolist(), [[0.0, 0.0], [2.0, 2.0]])

    def test_missing_query_token_raises(self):
        with self.assertRaises(ValueError):
            _gather_query_hidden(self.hidden, self.input_ids, token_id=999)

    def test_missing_doc_token_raises(self):
        with self.assertRaises(ValueError):
            _gather_doc_hiddens(self.hidden, self.input_ids, token_id=999)


class CosineScoresTest(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        query = mx.array([[1.0, 0.0, 0.0]])
        docs = mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        scores = _cosine_scores(query, docs)
        mx.eval(scores)
        self.assertAlmostEqual(scores.tolist()[0], 1.0, places=5)
        self.assertAlmostEqual(scores.tolist()[1], 1.0, places=5)

    def test_orthogonal_vectors_score_zero(self):
        query = mx.array([[1.0, 0.0]])
        docs = mx.array([[0.0, 1.0]])
        scores = _cosine_scores(query, docs)
        mx.eval(scores)
        self.assertAlmostEqual(abs(scores.tolist()[0]), 0.0, places=5)

    def test_opposite_vectors_score_negative(self):
        query = mx.array([[1.0, 0.0]])
        docs = mx.array([[-1.0, 0.0]])
        scores = _cosine_scores(query, docs)
        mx.eval(scores)
        self.assertAlmostEqual(scores.tolist()[0], -1.0, places=5)


class MLPProjectorTest(unittest.TestCase):
    def test_output_shape_is_512_for_1024_input(self):
        projector = MLPProjector()
        out = projector(mx.zeros((3, 1024)))
        self.assertEqual(out.shape, (3, 512))


class MLXRerankServiceTest(unittest.TestCase):
    def test_model_id_is_jina_v3(self):
        self.assertEqual(RERANK_MODEL_ID, "jinaai/jina-reranker-v3-mlx")

    def test_default_reranker_dir_matches_repo_layout(self):
        self.assertEqual(
            DEFAULT_RERANKER_DIR,
            PROJECT_DIR / "models" / "jinaai" / "jina-reranker-v3-mlx",
        )

    def test_service_exposes_jina_model_id(self):
        service = MLXRerankService(raw_reranker=FakeRawReranker())
        self.assertEqual(service.model_id, "jinaai/jina-reranker-v3-mlx")

    def test_normalizes_rerank_results(self):
        service = MLXRerankService(
            raw_reranker=FakeRawReranker(index=1, score=0.9),
            token_counter=lambda query, docs: 7,
        )
        result = service.rerank("question", ["first", "second"], top_n=1)
        self.assertEqual(result.total_tokens, 7)
        self.assertEqual(result.results[0].index, 1)
        self.assertEqual(result.results[0].relevance_score, 0.9)

    def test_service_passes_through_whatever_raw_returns_for_embedding(self):
        # The service itself does not synthesize embeddings; it only normalizes
        # whatever the raw reranker returns. JinaV3Reranker always returns None
        # (covered in JinaV3RerankerUnitTest). Here we verify the service path
        # is a faithful passthrough using a fake that returns a vector.
        service = MLXRerankService(raw_reranker=FakeRawReranker(index=0, embedding=[0.1, 0.2]))
        result = service.rerank("q", ["a"], top_n=1, return_embeddings=True)
        self.assertEqual(result.results[0].embedding, [0.1, 0.2])

    def test_default_token_counter_counts_words(self):
        service = MLXRerankService(raw_reranker=FakeRawReranker())
        result = service.rerank(
            "What is MLX?",
            ["MLX is an array framework.", "Another doc."],
            top_n=1,
        )
        self.assertEqual(result.total_tokens, 10)

    def test_rerank_clears_mlx_cache_after_inference(self):
        service = MLXRerankService(
            raw_reranker=FakeRawReranker(index=0),
            token_counter=lambda query, docs: 3,
        )
        with (
            patch("jina_v5_mlx_demo.reranking.mx.clear_cache") as clear_cache,
            patch("jina_v5_mlx_demo.reranking.mx.synchronize") as synchronize,
        ):
            service.rerank("query", ["doc"], top_n=1)
        clear_cache.assert_called_once_with()
        synchronize.assert_called_once_with()


class JinaV3RerankerUnitTest(unittest.TestCase):
    """Drives JinaV3Reranker.rerank with a fake backbone to assert end-to-end wiring
    without downloading real weights."""

    def _build_reranker(self, hidden_rows, input_ids, doc_indices, query_index, scores_expected):
        reranker = object.__new__(JinaV3Reranker)
        reranker.model_dir = PROJECT_DIR
        reranker.model = MagicMock()
        # model.model([input_ids])[0] -> hidden states [seq_len, hidden]
        hidden = mx.array(hidden_rows)
        reranker.model.model.return_value = [hidden]
        reranker.tokenizer = MagicMock()
        reranker.tokenizer.encode.return_value = input_ids
        reranker.projector = lambda x: x  # identity, so scores are pure cosine of hiddens
        reranker._query_token_id = input_ids[query_index]
        reranker._doc_token_id = input_ids[doc_indices[0]]
        return reranker

    def test_rerank_orders_documents_by_cosine_and_returns_null_embedding(self):
        # 3 hidden rows: doc0 at row0, doc1 at row1, query at row2
        # query direction = [1,0]; doc0 = [1,0] (score ~1), doc1 = [0,1] (score ~0)
        hidden_rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        input_ids = [100, 101, 102]  # 100=doc token, 102=query token
        reranker = self._build_reranker(
            hidden_rows, input_ids,
            doc_indices=[0], query_index=2, scores_expected=None,
        )
        results = reranker.rerank("q", ["doc one", "doc two"], top_n=2)

        # built prompt is whatever the mocked tokenizer was called with
        reranker.tokenizer.encode.assert_called_once()
        self.assertEqual(len(results), 2)
        # doc one (index 0, score ~1) ranks first
        self.assertEqual(results[0]["index"], 0)
        self.assertGreater(results[0]["relevance_score"], results[1]["relevance_score"])
        # embeddings always None
        self.assertTrue(all(r["embedding"] is None for r in results))
        # scores are cosine values in [-1, 1]
        for r in results:
            self.assertGreaterEqual(r["relevance_score"], -1.0)
            self.assertLessEqual(r["relevance_score"], 1.0)

    def test_top_n_truncates_results(self):
        hidden_rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        input_ids = [100, 101, 102]
        reranker = self._build_reranker(
            hidden_rows, input_ids, doc_indices=[0], query_index=2, scores_expected=None,
        )
        results = reranker.rerank("q", ["a", "b"], top_n=1)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
