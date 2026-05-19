import unittest

from fastapi.testclient import TestClient

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


if __name__ == "__main__":
    unittest.main()
