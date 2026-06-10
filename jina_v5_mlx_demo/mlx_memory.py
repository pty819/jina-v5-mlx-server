import threading

import mlx.core as mx


def _mb_to_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("MLX memory limits must be >= 0")
    return value * 1024 * 1024


class MLXMemoryManager:
    def __init__(
        self,
        *,
        cache_limit_mb: int | None = None,
        memory_limit_mb: int | None = None,
        trim_cache_when_idle: bool = True,
    ):
        self.cache_limit_mb = cache_limit_mb
        self.memory_limit_mb = memory_limit_mb
        self.trim_cache_when_idle = trim_cache_when_idle
        self._lock = threading.Lock()
        self._configured = False

    def configure(self):
        with self._lock:
            if self._configured:
                return
            memory_limit = _mb_to_bytes(self.memory_limit_mb)
            cache_limit = _mb_to_bytes(self.cache_limit_mb)
            if memory_limit is not None:
                mx.set_memory_limit(memory_limit)
            if cache_limit is not None:
                mx.set_cache_limit(cache_limit)
            self._configured = True

    def trim_if_idle(self, queue_state: dict[str, int]):
        if not self.trim_cache_when_idle:
            return
        if queue_state.get("unfinished", 0) != 0:
            return
        with self._lock:
            mx.clear_cache()
            mx.synchronize()

    def snapshot(self) -> dict:
        result = {
            "active_mb": round(mx.get_active_memory() / 1024**2),
            "cache_mb": round(mx.get_cache_memory() / 1024**2),
        }
        if self.cache_limit_mb is not None:
            result["cache_limit_mb"] = self.cache_limit_mb
        if self.memory_limit_mb is not None:
            result["memory_limit_mb"] = self.memory_limit_mb
        result["trim_cache_when_idle"] = self.trim_cache_when_idle
        return result
