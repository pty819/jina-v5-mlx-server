import unittest
from unittest.mock import patch

from jina_v5_mlx_demo.mlx_memory import MLXMemoryManager


class MLXMemoryManagerTest(unittest.TestCase):
    def test_configures_cache_and_memory_limits_in_bytes(self):
        manager = MLXMemoryManager(cache_limit_mb=1024, memory_limit_mb=4096)

        with (
            patch("jina_v5_mlx_demo.mlx_memory.mx.set_cache_limit") as set_cache_limit,
            patch("jina_v5_mlx_demo.mlx_memory.mx.set_memory_limit") as set_memory_limit,
        ):
            manager.configure()
            manager.configure()

        set_cache_limit.assert_called_once_with(1024 * 1024 * 1024)
        set_memory_limit.assert_called_once_with(4096 * 1024 * 1024)

    def test_trim_if_idle_clears_cache_only_when_no_unfinished_work(self):
        manager = MLXMemoryManager(cache_limit_mb=1024)

        with (
            patch("jina_v5_mlx_demo.mlx_memory.mx.clear_cache") as clear_cache,
            patch("jina_v5_mlx_demo.mlx_memory.mx.synchronize") as synchronize,
        ):
            manager.trim_if_idle({"unfinished": 1})
            manager.trim_if_idle({"unfinished": 0})

        clear_cache.assert_called_once_with()
        synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
