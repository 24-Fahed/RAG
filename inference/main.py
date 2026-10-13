"""Inference worker FastAPI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--mode",
    default="local",
    choices=["local", "staging", "production"],
    help="Deployment mode: local / staging / production (default: local)",
)
_args, _ = _parser.parse_known_args()

from inference.config import load_config
from rag_langgraph.indexing.embedding import get_embedder
from rag_langgraph.models.compressors import get_compressor
from rag_langgraph.models.generator import get_generator
from rag_langgraph.models.rerankers import get_reranker
from runtime_diagnostics import install_runtime_diagnostics

load_config(_args.mode)

from inference.config import EMBEDDING_MODEL, HOST, LLM_MODEL_PATH, MODE, PORT

logger = logging.getLogger("inference.preload")

DEFAULT_RERANK_MODEL = "monot5"
DEFAULT_COMPRESSION_METHOD = "recomp_extractive"

app = FastAPI(title="RAG Inference Worker", version="0.1.0")
install_runtime_diagnostics(app, service_name="inference", mode=MODE)

app.state.ready = MODE == "mock"
app.state.preloading = MODE != "mock"
app.state.preload_error = None
app.state.preloaded_components = []
app.state.preload_started_at = None
app.state.preload_completed_at = None


def _preload_component(name: str, loader) -> str:
    started_at = time.time()
    loader()
    elapsed = time.time() - started_at
    logger.info("preloaded component=%s elapsed=%.2fs", name, elapsed)
    return name


def _warmup_models() -> None:
    app.state.preload_started_at = time.time()
    app.state.preloading = True
    app.state.preload_error = None
    app.state.preloaded_components = []

    try:
        components = []

        if LLM_MODEL_PATH:
            components.append(("generator", lambda: get_generator(model_path=LLM_MODEL_PATH)))

        if EMBEDDING_MODEL:
            components.append(("embedder", lambda: get_embedder(model_name=EMBEDDING_MODEL)))

        components.append(("reranker", lambda: get_reranker(DEFAULT_RERANK_MODEL)))
        components.append(("compressor", lambda: get_compressor(DEFAULT_COMPRESSION_METHOD)))

        for name, loader in components:
            loaded_name = _preload_component(name, loader)
            app.state.preloaded_components.append(loaded_name)

        app.state.ready = True
        logger.info(
            "inference worker ready preloaded_components=%s elapsed=%.2fs",
            app.state.preloaded_components,
            time.time() - app.state.preload_started_at,
        )
    except Exception as exc:
        app.state.preload_error = str(exc)
        logger.exception("model preload failed")
    finally:
        app.state.preloading = False
        app.state.preload_completed_at = time.time()


@app.on_event("startup")
async def startup_preload() -> None:
    if MODE == "mock" or os.environ.get("RAG_PRELOAD_MODELS", "0") != "1":
        app.state.ready = True
        app.state.preloading = False
        return
    thread = threading.Thread(target=_warmup_models, name="model-preload", daemon=True)
    thread.start()


@app.middleware("http")
async def readiness_middleware(request: Request, call_next):
    if request.url.path.startswith("/inference") and not app.state.ready:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Inference worker is warming up",
                "ready": app.state.ready,
                "preloading": app.state.preloading,
                "error": app.state.preload_error,
                "preloaded_components": app.state.preloaded_components,
            },
        )
    return await call_next(request)


if MODE == "mock":
    from inference.mock_routers import router

    app.include_router(router, prefix="/inference")
else:
    from inference.routers import classify, compress, embed, generate, rerank

    app.include_router(classify.router, prefix="/inference", tags=["classify"])
    app.include_router(embed.router, prefix="/inference", tags=["embed"])
    app.include_router(rerank.router, prefix="/inference", tags=["rerank"])
    app.include_router(compress.router, prefix="/inference", tags=["compress"])
    app.include_router(generate.router, prefix="/inference", tags=["hyde"])


@app.exception_handler(ValueError)
async def invalid_model(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def model_failure(request, exc):
    return JSONResponse(status_code=502, content={"detail": "Model initialization or inference failed"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "ready": app.state.ready,
        "preloading": app.state.preloading,
        "error": app.state.preload_error,
        "preloaded_components": app.state.preloaded_components,
        "preload_started_at": app.state.preload_started_at,
        "preload_completed_at": app.state.preload_completed_at,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    print(f"Inference Worker starting in [{MODE}] mode at {HOST}:{PORT}")
    print(f"Config file: config/{_args.mode}.yaml")
    print(f"PID={os.getpid()} PPID={os.getppid()}")
    uvicorn.run(app, host=HOST, port=PORT)
