import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main


class ServeDefaultsTest(unittest.TestCase):
    def test_serve_defaults_to_twenty_minute_idle_unload_and_zero_cache(self):
        with (
            patch("sys.argv", ["main.py", "serve"]),
            patch("main.MLXEmbeddingService") as embedding_service,
            patch("main.OfficialMLXRerankService") as rerank_service,
            patch("main.ChatProxyClient") as chat_proxy,
            patch("main.create_app") as create_app,
            patch("main.uvicorn.run"),
        ):
            embedding_service.return_value.model_id = "embed"
            rerank_service.return_value.model_id = "rerank"
            chat_proxy.return_value.model_id = "chat"
            chat_proxy.return_value.upstream_base_url = "http://127.0.0.1:8001/v1"
            chat_proxy.return_value.upstream_model = "chat"
            create_app.return_value = Mock()

            main.main()

        embedding_service.assert_called_once_with(Path(main.DEFAULT_MODEL_DIR), idle_seconds=20 * 60)
        rerank_service.assert_called_once_with(Path(main.DEFAULT_RERANKER_DIR), idle_seconds=20 * 60)
        self.assertEqual(create_app.call_args.kwargs["mlx_cache_limit_mb"], 0)


if __name__ == "__main__":
    unittest.main()
