import unittest

from jina_v5_mlx_demo.metrics import RequestMetrics


class RequestMetricsTest(unittest.TestCase):
    def test_counts_rolling_windows_by_workload(self):
        now = [1_000_000.0]
        metrics = RequestMetrics(clock=lambda: now[0])

        metrics.record("embedding")
        metrics.record("rerank")
        now[0] += 3_601
        metrics.record("rerank")

        snapshot = metrics.snapshot(
            embedding_state={"queued": 1, "active": 2, "unfinished": 3},
            rerank_state={"queued": 0, "active": 1, "unfinished": 1},
        )

        self.assertEqual(snapshot["embedding"]["requests_1h"], 0)
        self.assertEqual(snapshot["embedding"]["requests_1d"], 1)
        self.assertEqual(snapshot["rerank"]["requests_1h"], 1)
        self.assertEqual(snapshot["rerank"]["requests_1d"], 2)
        self.assertEqual(snapshot["embedding"]["unfinished"], 3)

    def test_fresh_store_resets_counts(self):
        metrics = RequestMetrics(clock=lambda: 1_000.0)
        metrics.record("embedding")
        fresh = RequestMetrics(clock=lambda: 1_000.0)
        snapshot = fresh.snapshot(
            embedding_state={"queued": 0, "active": 0, "unfinished": 0},
            rerank_state={"queued": 0, "active": 0, "unfinished": 0},
        )
        self.assertEqual(snapshot["embedding"]["requests_1d"], 0)


if __name__ == "__main__":
    unittest.main()
