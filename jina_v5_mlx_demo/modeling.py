import gc
import importlib.util
import json
import threading
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


MODEL_ID = "jina-embeddings-v5-text-small"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "jina-embeddings-v5-text-small-retrieval-mlx"

DEFAULT_IDLE_SECONDS = 20 * 60


def _clear_mlx_cache():
    mx.clear_cache()
    mx.synchronize()


class MLXEmbeddingService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        *,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
        clear_cache_after_inference: bool = True,
    ):
        self.model_id = MODEL_ID
        self.model_dir = model_dir
        self.clear_cache_after_inference = clear_cache_after_inference
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._evictor = IdleEvictor(
            evict=self._evict_model,
            idle_seconds=idle_seconds,
        )

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str,
        dimensions: int,
        max_length: int = 8192,
    ) -> list[list[float]]:
        model, tokenizer = self._load()
        self._evictor.touch()
        try:
            with self._encode_lock:
                embeddings = model.encode(
                    texts,
                    tokenizer,
                    task_type=task_type,
                    truncate_dim=dimensions,
                    max_length=max_length,
                )
                mx.eval(embeddings)
                return embeddings.tolist()
        finally:
            if self.clear_cache_after_inference:
                _clear_mlx_cache()

    def count_tokens(self, texts: list[str], task_type: str) -> int:
        _, tokenizer = self._load()
        self._evictor.touch()
        prefix = {
            "retrieval.query": "Query: ",
            "retrieval.passage": "Document: ",
            "classification": "Document: ",
            "text-matching": "Document: ",
            "clustering": "Document: ",
        }.get(task_type, "")
        return sum(len(tokenizer.encode(prefix + text).ids) for text in texts)

    def _evict_model(self):
        with self._load_lock:
            if self._model is not None:
                self._model = None
                self._tokenizer = None
                gc.collect()
                _clear_mlx_cache()

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            model_dir = self.model_dir
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_dir}. Run: "
                    "uv run hf download jinaai/jina-embeddings-v5-text-small-retrieval-mlx "
                    "--local-dir models/jina-embeddings-v5-text-small-retrieval-mlx"
                )

            spec = importlib.util.spec_from_file_location("jina_mlx_model", model_dir / "model.py")
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load model implementation from {model_dir}")

            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            with (model_dir / "config.json").open() as f:
                config = json.load(f)
            # ``quantization`` is metadata for MLX loaders, not a model arg.
            quant_cfg = config.pop("quantization", None)

            model = model_module.JinaEmbeddingModel(config)
            weights = mx.load(str(model_dir / "model.safetensors"))
            if not isinstance(weights, dict):
                raise RuntimeError(f"Expected safetensors dict from {model_dir / 'model.safetensors'}")
            # If the weights are quantized, swap Linear -> QuantizedLinear so the
            # packed weight/scales/biases keys line up with the model structure.
            # Only layers that were actually quantized (i.e. have ``.scales`` in
            # the weight file) are converted — matching mlx-lm's loader, so an
            # nvfp4 group_size of 16 etc. don't trip affine-only validation on
            # unquantized layers. Embedding stays in fp16.
            if quant_cfg is not None:
                import mlx.nn as nn
                weight_keys = set(weights.keys())

                def _quant_predicate(path: str, module) -> bool:
                    if not isinstance(module, nn.Linear):
                        return False
                    return f"{path}.scales" in weight_keys

                nn.quantize(
                    model,
                    group_size=quant_cfg.get("group_size", 64),
                    bits=quant_cfg.get("bits", 8),
                    mode=quant_cfg.get("mode", "affine"),
                    class_predicate=_quant_predicate,
                )
            model.load_weights(list(weights.items()))

            self._model = model
            self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

            self._evictor.start()
            return self._model, self._tokenizer
