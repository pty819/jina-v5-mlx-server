from collections import deque
import time


HOUR = 60 * 60
DAY = 24 * HOUR


class RequestMetrics:
    def __init__(self, clock=time.time):
        self.clock = clock
        self._events: dict[str, deque[float]] = {
            "embedding": deque(),
            "rerank": deque(),
            "chat": deque(),
        }

    def record(self, workload: str) -> None:
        if workload not in self._events:
            raise ValueError(f"Unknown workload: {workload}")
        now = self.clock()
        self._events[workload].append(now)
        self._prune(workload, now)

    def snapshot(self, *, embedding_state: dict, rerank_state: dict, chat_state: dict | None = None) -> dict:
        now = self.clock()
        return {
            "embedding": self._workload_snapshot("embedding", now, embedding_state),
            "rerank": self._workload_snapshot("rerank", now, rerank_state),
            "chat": self._workload_snapshot(
                "chat",
                now,
                chat_state or {"queued": 0, "active": 0, "unfinished": 0},
            ),
        }

    def _prune(self, workload: str, now: float) -> None:
        cutoff = now - DAY
        events = self._events[workload]
        while events and events[0] < cutoff:
            events.popleft()

    def _workload_snapshot(self, workload: str, now: float, state: dict) -> dict:
        events = self._events[workload]
        cutoff_1h = now - HOUR
        cutoff_1d = now - DAY
        requests_1h = sum(1 for ts in events if ts >= cutoff_1h)
        requests_1d = sum(1 for ts in events if ts >= cutoff_1d)
        return {
            "requests_1h": requests_1h,
            "requests_1d": requests_1d,
            "queued": state["queued"],
            "active": state["active"],
            "unfinished": state["unfinished"],
        }
