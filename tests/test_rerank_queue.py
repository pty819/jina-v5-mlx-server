import asyncio
import threading
import time
import unittest

from jina_v5_mlx_demo.batching import DynamicBatcher
from jina_v5_mlx_demo.reranking import RerankResponse, RerankResult
from jina_v5_mlx_demo.rerank_queue import RerankQueue


class InferenceTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.overlap = False

    def enter(self):
        with self.lock:
            self.active += 1
            self.overlap = self.overlap or self.active > 1

    def exit(self):
        with self.lock:
            self.active -= 1


class FakeRerankService:
    model_id = "fake-rerank-model"

    def rerank(self, query, documents, *, top_n, return_embeddings):
        return RerankResponse(
            results=[
                RerankResult(
                    index=0,
                    relevance_score=0.9,
                    document=documents[0],
                    embedding=[0.1, 0.2] if return_embeddings else None,
                ),
            ],
            total_tokens=len(query.split()) + sum(len(d.split()) for d in documents),
        )


class BlockingRerankService:
    model_id = "fake-rerank-model"

    def __init__(self, started, release):
        self.started = started
        self.release = release
        self._call_count = 0

    def rerank(self, query, documents, *, top_n, return_embeddings):
        self._call_count += 1
        self.started.set()
        self.release.wait(timeout=1)
        return RerankResponse(
            results=[
                RerankResult(index=0, relevance_score=0.5, document=documents[0]),
            ],
            total_tokens=1,
        )


class FailingRerankService:
    model_id = "fake-rerank-model"

    def rerank(self, query, documents, *, top_n, return_embeddings):
        raise RuntimeError("model crash")


class RerankQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_result_to_original_request(self):
        queue = RerankQueue(FakeRerankService(), inference_gate=asyncio.Lock())
        await queue.start()
        result = await queue.rerank(
            "query",
            ["doc"],
            top_n=1,
            return_embeddings=False,
        )
        self.assertEqual(result.results[0].document, "doc")
        self.assertEqual(queue.queue_state()["unfinished"], 0)
        await queue.stop()

    async def test_propagates_service_exception_to_caller(self):
        queue = RerankQueue(FailingRerankService(), inference_gate=asyncio.Lock())
        await queue.start()
        with self.assertRaises(RuntimeError):
            await queue.rerank("q", ["d"], top_n=1, return_embeddings=False)
        await queue.stop()

    async def test_reports_active_while_processing(self):
        started = threading.Event()
        release = threading.Event()
        queue = RerankQueue(
            BlockingRerankService(started, release),
            inference_gate=asyncio.Lock(),
        )
        await queue.start()
        task = asyncio.create_task(
            queue.rerank("q", ["d"], top_n=1, return_embeddings=False)
        )
        await asyncio.to_thread(started.wait, 1)
        self.assertEqual(queue.queue_state()["active"], 1)
        self.assertEqual(queue.queue_state()["unfinished"], 1)

        release.set()
        await task
        self.assertEqual(queue.queue_state()["unfinished"], 0)
        await queue.stop()


class SharedGateNonOverlapTest(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_and_rerank_do_not_overlap(self):
        tracker = InferenceTracker()

        class TrackedEmbeddingService:
            model_id = "fake-embed"

            def count_tokens(self, texts, task_type):
                return 1

            def embed(self, texts, *, task_type, dimensions, max_length):
                tracker.enter()
                time.sleep(0.05)
                tracker.exit()
                return [[1.0, 0.5] for _ in texts]

        class TrackedRerankService:
            model_id = "fake-rerank"

            def rerank(self, query, documents, *, top_n, return_embeddings):
                tracker.enter()
                time.sleep(0.05)
                tracker.exit()
                return RerankResponse(
                    results=[RerankResult(index=0, relevance_score=0.9, document=documents[0])],
                    total_tokens=1,
                )

        gate = asyncio.Lock()
        batcher = DynamicBatcher(
            TrackedEmbeddingService(),
            max_batch_size=1,
            batch_timeout_ms=0,
            inference_gate=gate,
        )
        rerank_queue = RerankQueue(TrackedRerankService(), inference_gate=gate)

        await batcher.start()
        await rerank_queue.start()

        embed_task = asyncio.create_task(
            batcher.embed(["t"], task_type="retrieval.passage", dimensions=32, max_length=8192)
        )
        rerank_task = asyncio.create_task(
            rerank_queue.rerank("q", ["d"], top_n=1, return_embeddings=False)
        )

        await asyncio.gather(embed_task, rerank_task)

        await rerank_queue.stop()
        await batcher.stop()

        self.assertFalse(tracker.overlap)


if __name__ == "__main__":
    unittest.main()
