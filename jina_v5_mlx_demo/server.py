import asyncio
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from jina_v5_mlx_demo.batching import DynamicBatcher
from jina_v5_mlx_demo.chat_queue import ChatQueue
from jina_v5_mlx_demo.metrics import RequestMetrics
from jina_v5_mlx_demo.rerank_queue import RerankQueue
from jina_v5_mlx_demo.routes import (
    register_chat_routes,
    register_embedding_routes,
    register_rerank_routes,
)
from jina_v5_mlx_demo.schema import RequestError, parse_max_length


def create_app(
    embedding_service,
    *,
    rerank_service=None,
    chat_service=None,
    metrics=None,
    max_batch_size: int = 4,
    batch_timeout_ms: int = 5,
    max_batch_tokens: int = 8192,
    length_tolerance: float = 0.2,
    default_max_length: int = 8192,
) -> FastAPI:
    parse_max_length(default_max_length)

    inference_gate = asyncio.Lock()
    batcher = DynamicBatcher(
        embedding_service,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        max_batch_tokens=max_batch_tokens,
        length_tolerance=length_tolerance,
        inference_gate=inference_gate,
    )

    rerank_queue = (
        RerankQueue(rerank_service, inference_gate=inference_gate)
        if rerank_service is not None
        else None
    )
    chat_queue = (
        ChatQueue(chat_service, inference_gate=inference_gate)
        if chat_service is not None
        else None
    )
    metrics = metrics or RequestMetrics()

    openai_router = APIRouter(prefix="/openai/v1")
    jina_router = APIRouter(prefix="/jina/v1")
    utils_router = APIRouter()

    # --- OpenAI group ---
    register_embedding_routes(openai_router, embedding_service, batcher, metrics, default_max_length, tags=["OpenAI"])
    if rerank_service is not None:
        register_rerank_routes(openai_router, rerank_service, rerank_queue, metrics, tags=["OpenAI"])
    if chat_service is not None:
        register_chat_routes(openai_router, chat_service, chat_queue, tags=["OpenAI"])

    # --- Jina group ---
    register_embedding_routes(jina_router, embedding_service, batcher, metrics, default_max_length, tags=["Jina"])
    if rerank_service is not None:
        register_rerank_routes(jina_router, rerank_service, rerank_queue, metrics, tags=["Jina"])

    # --- /v1 bare routes (same handlers, no prefix) ---
    v1_router = APIRouter(prefix="/v1")
    register_embedding_routes(v1_router, embedding_service, batcher, metrics, default_max_length, tags=["v1"])
    if rerank_service is not None:
        register_rerank_routes(v1_router, rerank_service, rerank_queue, metrics, tags=["v1"])
    if chat_service is not None:
        register_chat_routes(v1_router, chat_service, chat_queue, tags=["v1"])

    # --- Utils group ---
    @utils_router.get("/health", tags=["Utils"], summary="Health check")
    async def health():
        result = {"status": "ok", "embedding_model": embedding_service.model_id}
        if rerank_service is not None:
            result["rerank_model"] = rerank_service.model_id
        else:
            result["model"] = embedding_service.model_id
        if chat_service is not None:
            result["chat_model"] = chat_service.model_id
        return result

    @utils_router.get("/stats.json", tags=["Utils"], summary="Stats JSON")
    async def stats_json():
        snapshot = metrics.snapshot(
            embedding_state=batcher.queue_state(),
            rerank_state=rerank_queue.queue_state() if rerank_queue else {"queued": 0, "active": 0, "unfinished": 0},
        )
        snapshot["mlx_memory"] = {
            "active_mb": round(mx.get_active_memory() / 1024**2),
            "cache_mb": round(mx.get_cache_memory() / 1024**2),
        }
        return snapshot

    @utils_router.get("/stats", tags=["Utils"], summary="Stats dashboard")
    async def stats_html():
        snapshot = await stats_json()
        return HTMLResponse(content=_render_stats_html(snapshot))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await batcher.start()
        if rerank_queue is not None:
            await rerank_queue.start()
        if chat_queue is not None:
            await chat_queue.start()
        try:
            yield
        finally:
            if chat_queue is not None:
                await chat_queue.stop()
            if rerank_queue is not None:
                await rerank_queue.stop()
            await batcher.stop()
            if chat_service is not None and hasattr(chat_service, "close"):
                chat_service.close()

    app = FastAPI(
        title="Jina v5 MLX Server",
        version="0.2.0",
        description="Local FastAPI serving for Jina v5 embeddings and reranking.",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "OpenAI", "description": "OpenAI-compatible endpoints (`/openai/v1`)."},
            {"name": "Jina", "description": "Jina-native endpoints (`/jina/v1`)."},
            {"name": "v1", "description": "Bare `/v1` prefix — same handlers, no vendor group."},
            {"name": "Utils", "description": "Health, stats, and operator tools."},
        ],
    )

    app.exception_handler(RequestError)(request_error_handler)
    app.exception_handler(ValueError)(value_error_handler)
    app.exception_handler(RequestValidationError)(validation_error_handler)

    app.include_router(v1_router)
    app.include_router(openai_router)
    app.include_router(jina_router)
    app.include_router(utils_router)

    return app


async def request_error_handler(_request: Request, error: RequestError):
    return error_response(error.status, str(error))


async def value_error_handler(_request: Request, error: ValueError):
    return error_response(400, str(error))


async def validation_error_handler(_request: Request, error: RequestValidationError):
    return error_response(400, validation_message(error))


def _render_stats_html(snapshot: dict) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jina MLX Server Stats</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; background:#0f1117; color:#e0e0e0; padding:24px; }
  h1 { font-size:20px; font-weight:600; margin-bottom:4px; color:#fff; }
  .ts { font-size:12px; color:#888; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:720px; }
  .card { background:#1a1d27; border:1px solid #2a2d37; border-radius:10px; padding:16px 20px; }
  .card h2 { font-size:14px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }
  .card:nth-child(1) h2 { color:#6dd5fa; }
  .card:nth-child(2) h2 { color:#f6d365; }
  .metric { display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:13px; }
  .label { width:90px; color:#999; flex-shrink:0; }
  .value { font-weight:600; min-width:36px; text-align:right; }
  .value.num { font-variant-numeric:tabular-nums; font-size:16px; color:#fff; }
  .bar-track { flex:1; height:6px; background:#2a2d37; border-radius:3px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:3px; transition:width .4s ease; }
  .bar-fill.req { background:linear-gradient(90deg,#6dd5fa,#6dd5fa); }
  .card:nth-child(2) .bar-fill.req { background:linear-gradient(90deg,#f6d365,#f6d365); }
  .bar-fill.q { background:linear-gradient(90deg,#ff6b6b,#ee5a24); }
  .mlx { margin-top:16px; max-width:720px; background:#1a1d27; border:1px solid #2a2d37; border-radius:10px; padding:16px 20px; }
  .mlx h2 { font-size:14px; font-weight:600; color:#a29bfe; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
  .mlx .metric { margin-bottom:4px; }
  .foot { margin-top:12px; font-size:11px; color:#555; max-width:720px; display:flex; justify-content:space-between; }
  .dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:#4cd137; margin-right:6px; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .dot.err { background:#ff6b6b; animation:none; }
</style>
</head>
<body>
<h1>Jina MLX Server</h1>
<div class="ts" id="ts"></div>
<div class="grid">
  <div class="card" id="emb-card">
    <h2>Embedding</h2>
    <div class="metric"><span class="label">Requests 1h</span><div class="bar-track"><div class="bar-fill req" id="emb-1h-bar"></div></div><span class="value" id="emb-1h">-</span></div>
    <div class="metric"><span class="label">Requests 24h</span><div class="bar-track"><div class="bar-fill req" id="emb-1d-bar"></div></div><span class="value" id="emb-1d">-</span></div>
    <div class="metric"><span class="label">Queued</span><span class="value num" id="emb-q">-</span></div>
    <div class="metric"><span class="label">Active</span><span class="value num" id="emb-a">-</span></div>
    <div class="metric"><span class="label">Unfinished</span><div class="bar-track"><div class="bar-fill q" id="emb-uf-bar"></div></div><span class="value num" id="emb-uf">-</span></div>
  </div>
  <div class="card" id="rr-card">
    <h2>Rerank</h2>
    <div class="metric"><span class="label">Requests 1h</span><div class="bar-track"><div class="bar-fill req" id="rr-1h-bar"></div></div><span class="value" id="rr-1h">-</span></div>
    <div class="metric"><span class="label">Requests 24h</span><div class="bar-track"><div class="bar-fill req" id="rr-1d-bar"></div></div><span class="value" id="rr-1d">-</span></div>
    <div class="metric"><span class="label">Queued</span><span class="value num" id="rr-q">-</span></div>
    <div class="metric"><span class="label">Active</span><span class="value num" id="rr-a">-</span></div>
    <div class="metric"><span class="label">Unfinished</span><div class="bar-track"><div class="bar-fill q" id="rr-uf-bar"></div></div><span class="value num" id="rr-uf">-</span></div>
  </div>
</div>
<div class="mlx">
  <h2>MLX Memory</h2>
  <div class="metric"><span class="label">Active</span><span class="value num" id="mlx-active">-</span><span class="label" style="width:auto;color:#888">MB</span></div>
  <div class="metric"><span class="label">Cache</span><span class="value num" id="mlx-cache">-</span><span class="label" style="width:auto;color:#888">MB</span></div>
</div>
<div class="foot">
  <span><span class="dot" id="status-dot"></span><span id="status-text">connecting...</span></span>
  <span>Memory-only counters, reset on restart</span>
</div>
<script>
function $(id){return document.getElementById(id)}
function upd(prefix, s, d){
  $(prefix+'-1h').textContent=s.requests_1h;
  $(prefix+'-1d').textContent=s.requests_1d;
  $(prefix+'-q').textContent=s.queued;
  $(prefix+'-a').textContent=s.active;
  $(prefix+'-uf').textContent=s.unfinished;
  var pct=Math.max(s.requests_1d,1);
  $(prefix+'-1h-bar').style.width=(s.requests_1h/pct*100)+'%';
  $(prefix+'-1d-bar').style.width='100%';
  $(prefix+'-uf-bar').style.width=Math.min(s.unfinished*20,100)+'%';
}
function tick(){
  fetch('/stats.json').then(r=>r.json()).then(d=>{
    upd('emb',d.embedding);
    upd('rr',d.rerank);
    var m=d.mlx_memory||{};
    $('mlx-active').textContent=m.active_mb||0;
    $('mlx-cache').textContent=m.cache_mb||0;
    $('ts').textContent=new Date().toLocaleString();
    $('status-dot').className='dot';
    $('status-text').textContent='live';
  }).catch(()=>{
    $('status-dot').className='dot err';
    $('status-text').textContent='fetch error';
  });
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>"""


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
            }
        },
    )


def validation_message(error: RequestValidationError) -> str:
    errors = error.errors()
    if not errors:
        return "Invalid request"
    first_error = errors[0]
    context = first_error.get("ctx")
    if isinstance(context, dict) and context.get("error"):
        return str(context["error"])
    if first_error.get("loc") == ("body", "input", "str") or first_error.get("loc") == ("body", "input", "list[str]"):
        return "input must be a string or a list of strings"
    return str(first_error.get("msg", "Invalid request"))
