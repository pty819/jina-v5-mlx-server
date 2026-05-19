import asyncio
import unittest

from jina_v5_mlx_demo.batching import DynamicBatcher


class RecordingEmbeddingService:
    model_id = "fake-model"

    def __init__(self):
        self.batches = []

    def embed(self, texts, *, task_type, dimensions, max_length):
        self.batches.append(
            {
                "texts": texts,
                "task_type": task_type,
                "dimensions": dimensions,
                "max_length": max_length,
            }
        )
        return [[float(len(texts)), float(index)] for index, _ in enumerate(texts)]

    def count_tokens(self, texts, task_type):
        return sum(int(text) for text in texts)


class DynamicBatcherTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = RecordingEmbeddingService()
        self.batcher = DynamicBatcher(
            self.service,
            max_batch_size=4,
            batch_timeout_ms=10,
            max_batch_tokens=10_000,
            length_tolerance=0.2,
        )
        await self.batcher.start()

    async def asyncTearDown(self):
        await self.batcher.stop()

    async def test_groups_queued_items_by_similar_token_length(self):
        results = await asyncio.gather(
            self.batcher.embed(["100"], task_type="retrieval.passage", dimensions=32, max_length=8192),
            self.batcher.embed(["1000"], task_type="retrieval.passage", dimensions=32, max_length=8192),
            self.batcher.embed(["110"], task_type="retrieval.passage", dimensions=32, max_length=8192),
            self.batcher.embed(["950"], task_type="retrieval.passage", dimensions=32, max_length=8192),
            self.batcher.embed(["1050"], task_type="retrieval.passage", dimensions=32, max_length=8192),
        )

        self.assertEqual([result[0] for result in results], [[2.0, 0.0], [3.0, 0.0], [2.0, 1.0], [3.0, 1.0], [3.0, 2.0]])
        self.assertEqual([batch["texts"] for batch in self.service.batches], [["100", "110"], ["1000", "950", "1050"]])

    async def test_does_not_mix_incompatible_request_settings(self):
        await asyncio.gather(
            self.batcher.embed(["100"], task_type="retrieval.query", dimensions=32, max_length=8192),
            self.batcher.embed(["110"], task_type="retrieval.passage", dimensions=32, max_length=8192),
            self.batcher.embed(["115"], task_type="retrieval.query", dimensions=64, max_length=8192),
        )

        self.assertEqual(len(self.service.batches), 3)
        self.assertEqual({batch["task_type"] for batch in self.service.batches}, {"retrieval.query", "retrieval.passage"})
        self.assertEqual({batch["dimensions"] for batch in self.service.batches}, {32, 64})


if __name__ == "__main__":
    unittest.main()
