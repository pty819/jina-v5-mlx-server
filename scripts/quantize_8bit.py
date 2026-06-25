#!/usr/bin/env python3
"""Quantize the embedding and rerank models to reduced precision.

Supported recipes:
    8bit   : affine mode, group_size=64, bits=8. No calibration. Quantizes
             backbone Linear + projector Linear. Output suffix: ``-8bit``.
    nvfp4  : nvfp4 mode, group_size=16, bits=4 (E2M1 weights, E4M3 scale). No
             calibration. Quantizes backbone Linear only; the 3 MB projector is
             left in fp16 (its loading path stays simple and the size win is
             negligible). Output suffix: ``-nvfp4``.

Both recipes skip Embedding layers to preserve vocabulary fidelity.

Usage:
    uv run python scripts/quantize_8bit.py <8bit|nvfp4> <embedding|rerank|both>

Outputs:
    models/jina-embeddings-v5-text-small-retrieval-mlx-{8bit|nvfp4}/
    models/jina-reranker-v3-mlx-{8bit|nvfp4}/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow running from scripts/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.utils as nn_utils

PROJECT_DIR = Path(__file__).resolve().parent.parent

EMBED_SRC = PROJECT_DIR / "models" / "jina-embeddings-v5-text-small-retrieval-mlx"
RERANK_SRC = PROJECT_DIR / "models" / "jina-reranker-v3-mlx"

# (mode, group_size, bits, suffix, quantize_projector)
RECIPES = {
    "8bit":  {"mode": "affine", "group_size": 64,  "bits": 8, "suffix": "8bit",  "quantize_projector": True},
    "nvfp4": {"mode": "nvfp4",  "group_size": 16,  "bits": 4, "suffix": "nvfp4", "quantize_projector": False},
}

_SKIP_FILES = {"model.safetensors", "projector.safetensors", "model.safetensors.index.json",
               "rerank.py", "test_examples.py"}


def _linear_only(_path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def _copy_metadata(src: Path, dst: Path, skip: set[str]) -> None:
    for item in src.iterdir():
        if item.name in skip or item.name in {".cache", "__pycache__"}:
            continue
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)


def _write_quant_config(dst: Path, recipe: dict) -> None:
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization"] = {
        "group_size": recipe["group_size"],
        "bits": recipe["bits"],
        "mode": recipe["mode"],
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def _report(label: str, src: Path, dst: Path) -> None:
    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    pct = dst_mb / src_mb * 100 if src_mb else 0
    print(f"  [{label}] {src_mb:.1f} MB -> {dst_mb:.1f} MB ({pct:.0f}%)")


def quantize_embedding(recipe: dict) -> None:
    import importlib.util

    dst = EMBED_SRC.with_name(f"{EMBED_SRC.name}-{recipe['suffix']}")
    print(f"[embedding/{recipe['mode']}] source: {EMBED_SRC}")
    if not (EMBED_SRC / "model.safetensors").exists():
        sys.exit(f"missing {EMBED_SRC/'model.safetensors'}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    spec = importlib.util.spec_from_file_location("jina_mlx_model", EMBED_SRC / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((EMBED_SRC / "config.json").read_text())
    model = module.JinaEmbeddingModel(config)
    weights = mx.load(str(EMBED_SRC / "model.safetensors"))
    model.load_weights(list(weights.items()))

    print(f"[embedding] quantizing Linear layers -> {recipe['mode']} {recipe['bits']}bit/g{recipe['group_size']}")
    nn.quantize(
        model,
        group_size=recipe["group_size"], bits=recipe["bits"], mode=recipe["mode"],
        class_predicate=_linear_only,
    )
    params = dict(nn_utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(dst / "model.safetensors"), params)

    _copy_metadata(EMBED_SRC, dst, skip={"model.safetensors"})
    _write_quant_config(dst, recipe)
    _report("embedding", EMBED_SRC / "model.safetensors", dst / "model.safetensors")
    print(f"[embedding] done -> {dst}")


def quantize_rerank(recipe: dict) -> None:
    from mlx_lm import load

    dst = RERANK_SRC.with_name(f"{RERANK_SRC.name}-{recipe['suffix']}")
    print(f"[rerank/{recipe['mode']}] source: {RERANK_SRC}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    model, _tokenizer = load(str(RERANK_SRC))
    model.eval()
    print(f"[rerank] quantizing backbone Linear -> {recipe['mode']} {recipe['bits']}bit/g{recipe['group_size']}")
    nn.quantize(
        model,
        group_size=recipe["group_size"], bits=recipe["bits"], mode=recipe["mode"],
        class_predicate=_linear_only,
    )
    backbone_params = dict(nn_utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(dst / "model.safetensors"), backbone_params)

    if recipe["quantize_projector"]:
        # affine path: projector is a small MLP, quantize its Linears too so the
        # loading code builds QuantizedLinear shells for them.
        from safetensors import safe_open
        from jina_v5_mlx_demo.reranking import MLPProjector
        projector = MLPProjector()
        proj_weights = {}
        with safe_open(str(RERANK_SRC / "projector.safetensors"), framework="numpy") as f:
            for k in f.keys():
                proj_weights[k] = mx.array(f.get_tensor(k))
        projector.load_weights(list(proj_weights.items()))
        print(f"[rerank] quantizing projector -> {recipe['mode']}")
        nn.quantize(
            projector,
            group_size=recipe["group_size"], bits=recipe["bits"], mode=recipe["mode"],
            class_predicate=_linear_only,
        )
        mx.save_safetensors(str(dst / "projector.safetensors"),
                            dict(nn_utils.tree_flatten(projector.parameters())))
        _report("projector", RERANK_SRC / "projector.safetensors", dst / "projector.safetensors")
    else:
        # nvfp4 path: keep projector in fp16 — it is only 3 MB and avoiding a
        # separate nvfp4 loading branch keeps _build_projector_linear simple.
        shutil.copy2(RERANK_SRC / "projector.safetensors", dst / "projector.safetensors")
        print(f"[rerank] projector kept fp16 (copied as-is)")

    _copy_metadata(RERANK_SRC, dst, skip=_SKIP_FILES)
    _write_quant_config(dst, recipe)
    _report("backbone", RERANK_SRC / "model.safetensors", dst / "model.safetensors")
    print(f"[rerank] done -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=list(RECIPES.keys()))
    parser.add_argument("target", choices=["embedding", "rerank", "both"])
    args = parser.parse_args()
    recipe = RECIPES[args.recipe]
    if args.target in ("embedding", "both"):
        quantize_embedding(recipe)
    if args.target in ("rerank", "both"):
        quantize_rerank(recipe)


if __name__ == "__main__":
    main()
