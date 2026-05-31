import unittest

from fastapi.testclient import TestClient

from jina_v5_mlx_demo.reranking import RerankResponse, RerankResult
from jina_v5_mlx_demo.server import create_app


class FakeEmbeddingService:
    model_id = "fake-embedding-model"

    def embed(self, texts, *, task_type, dimensions, max_length):
        self.last_call = {
            "texts": texts,
            "task_type": task_type,
            "dimensions": dimensions,
            "max_length": max_length,
        }
        return [[float(len(text.split())), 0.5] for text in texts]

    def count_tokens(self, texts, task_type):
        return sum(len(text.split()) for text in texts)


class FakeRerankService:
    model_id = "jina-reranker-v3"

    def rerank(self, query, documents, *, top_n, return_embeddings):
        results = []
        for i, doc in enumerate(documents):
            results.append(
                RerankResult(
                    index=i,
                    relevance_score=0.9 - i * 0.1,
                    document=doc,
                    embedding=[0.1 * i, 0.2 * i] if return_embeddings else None,
                )
            )
        return RerankResponse(
            results=results[:top_n] if top_n else results,
            total_tokens=len(query.split()) + sum(len(d.split()) for d in documents),
        )


class FakeChatService:
    model_id = "fake-chat-model"

    def complete(self, messages, *, max_tokens, temperature, top_p, stop):
        self.last_call = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        return {
            "content": f"reply:{messages[-1]['content']}",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "finish_reason": "stop",
        }


class EmbeddingServerTest(unittest.TestCase):
    def setUp(self):
        self.service = FakeEmbeddingService()
        self.client = TestClient(create_app(self.service))

    def test_openai_style_embeddings_endpoint(self):
        response = self.client.post(
            "/openai/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": ["alpha beta", "gamma"],
                "dimensions": 32,
                "encoding_format": "float",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "fake-embedding-model")
        self.assertEqual(body["data"][0]["embedding"], [2.0, 0.5])
        self.assertEqual(body["data"][1]["embedding"], [1.0, 0.5])
        self.assertEqual(body["usage"]["prompt_tokens"], 3)
        self.assertEqual(self.service.last_call["task_type"], "retrieval.passage")
        self.assertEqual(self.service.last_call["dimensions"], 32)

    def test_jina_style_embeddings_endpoint(self):
        response = self.client.post(
            "/jina/v1/embeddings",
            json={
                "input": "alpha beta",
                "task": "retrieval.query",
                "dimensions": 64,
                "normalized": True,
                "embedding_type": "float",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["embedding"], [2.0, 0.5])
        self.assertEqual(body["usage"]["total_tokens"], 2)
        self.assertEqual(self.service.last_call["texts"], ["alpha beta"])
        self.assertEqual(self.service.last_call["task_type"], "retrieval.query")
        self.assertEqual(self.service.last_call["dimensions"], 64)

    def test_shared_v1_endpoint_accepts_jina_task_field(self):
        response = self.client.post(
            "/v1/embeddings",
            json={
                "input": ["alpha"],
                "task": "classification",
                "dimensions": 128,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.last_call["task_type"], "classification")

    def test_openai_style_endpoint_accepts_input_type(self):
        response = self.client.post(
            "/openai/v1/embeddings",
            json={
                "input": ["alpha"],
                "input_type": "query",
                "dimensions": 128,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.last_call["task_type"], "retrieval.query")

    def test_openai_style_endpoint_accepts_model_alias(self):
        response = self.client.post(
            "/openai/v1/embeddings",
            json={
                "model": "jina-v5-query",
                "input": ["alpha"],
                "dimensions": 128,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.last_call["task_type"], "retrieval.query")

    def test_rejects_invalid_input_shape(self):
        response = self.client.post("/v1/embeddings", json={"input": {"not": "valid"}})

        self.assertEqual(response.status_code, 400)
        self.assertIn("input must be a string or a list of strings", response.json()["error"]["message"])

    def test_openai_sdk_encoding_format_value_still_returns_float_arrays(self):
        response = self.client.post(
            "/openai/v1/embeddings",
            json={
                "input": ["alpha beta"],
                "dimensions": 32,
                "encoding_format": "base64",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["embedding"], [2.0, 0.5])

    def test_rejects_unsupported_jina_embedding_type(self):
        response = self.client.post(
            "/jina/v1/embeddings",
            json={
                "input": "alpha",
                "embedding_type": "base64",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only float embeddings are supported", response.json()["error"]["message"])


class CombinedServerTest(unittest.TestCase):
    def setUp(self):
        self.embedding_service = FakeEmbeddingService()
        self.rerank_service = FakeRerankService()
        self.chat_service = FakeChatService()
        self.client = TestClient(
            create_app(
                self.embedding_service,
                rerank_service=self.rerank_service,
                chat_service=self.chat_service,
            )
        )

    def test_jina_style_rerank_endpoint(self):
        response = self.client.post(
            "/jina/v1/rerank",
            json={
                "model": "jina-reranker-v3",
                "query": "question",
                "documents": ["first", "second"],
                "top_n": 1,
                "return_documents": True,
                "return_embeddings": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "jina-reranker-v3")
        self.assertEqual(body["usage"]["total_tokens"], 3)
        self.assertEqual(body["results"][0]["index"], 0)
        self.assertEqual(body["results"][0]["document"], "first")
        self.assertEqual(body["results"][0]["embedding"], [0.0, 0.0])

    def test_v1_rerank_endpoint(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "q",
                "documents": ["a", "b"],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["results"]), 2)
        self.assertIn("document", body["results"][0])

    def test_openai_compatible_rerank_alias(self):
        response = self.client.post(
            "/openai/v1/rerank",
            json={
                "query": "q",
                "documents": ["a", "b"],
            },
        )

        self.assertEqual(response.status_code, 200)

    def test_return_documents_false_omits_document(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "q",
                "documents": ["a", "b"],
                "return_documents": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("document", body["results"][0])

    def test_rejects_empty_documents(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "q",
                "documents": [],
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_query(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "  ",
                "documents": ["a"],
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_top_n_zero(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "q",
                "documents": ["a"],
                "top_n": 0,
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_unsupported_model(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "model": "unknown-model",
                "query": "q",
                "documents": ["a"],
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_extra_rerank_field(self):
        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "q",
                "documents": ["a"],
                "unexpected_field": True,
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_stats_json_splits_counts(self):
        self.client.post("/v1/embeddings", json={"input": "hello", "dimensions": 32})
        self.client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})

        response = self.client.get("/stats.json")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["embedding"]["requests_1d"], 1)
        self.assertEqual(body["rerank"]["requests_1d"], 1)

    def test_stats_html_renders(self):
        response = self.client.get("/stats")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Embedding", response.text)
        self.assertIn("Rerank", response.text)
        self.assertIn("Unfinished", response.text)

    def test_health_reports_both_models(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["embedding_model"], "fake-embedding-model")
        self.assertEqual(body["rerank_model"], "jina-reranker-v3")
        self.assertEqual(body["chat_model"], "fake-chat-model")

    def test_openai_chat_completions_endpoint(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-chat-model",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 32,
                "temperature": 0.2,
                "top_p": 0.9,
                "stop": ["END"],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "fake-chat-model")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(body["choices"][0]["message"]["content"], "reply:hello")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["prompt_tokens"], 5)
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        self.assertEqual(body["usage"]["total_tokens"], 8)
        self.assertEqual(self.chat_service.last_call["max_tokens"], 32)
        self.assertEqual(self.chat_service.last_call["temperature"], 0.2)
        self.assertEqual(self.chat_service.last_call["top_p"], 0.9)
        self.assertEqual(self.chat_service.last_call["stop"], ["END"])

    def test_openai_chat_alias_endpoint(self):
        response = self.client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "reply:hello")

    def test_rejects_streaming_chat_requests(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("stream=true is not supported", response.json()["error"]["message"])

    def test_create_app_does_not_reuse_registered_routes(self):
        first_app = create_app(
            FakeEmbeddingService(),
            rerank_service=FakeRerankService(),
            chat_service=FakeChatService(),
        )
        second_app = create_app(
            FakeEmbeddingService(),
            rerank_service=FakeRerankService(),
            chat_service=FakeChatService(),
        )

        first_paths = [getattr(route, "path", None) for route in first_app.routes]
        second_paths = [getattr(route, "path", None) for route in second_app.routes]

        self.assertEqual(first_paths.count("/openai/v1/embeddings"), 1)
        self.assertEqual(second_paths.count("/openai/v1/embeddings"), 1)
        self.assertEqual(first_paths.count("/openai/v1/rerank"), 1)
        self.assertEqual(second_paths.count("/openai/v1/rerank"), 1)
        self.assertEqual(first_paths.count("/openai/v1/chat/completions"), 1)
        self.assertEqual(second_paths.count("/openai/v1/chat/completions"), 1)
        self.assertEqual(first_paths.count("/health"), 1)
        self.assertEqual(second_paths.count("/health"), 1)


if __name__ == "__main__":
    unittest.main()
