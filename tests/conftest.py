"""Offline tests exercise real HTTP schemas and adapters without model downloads."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.config import load_config
load_config('local')


@pytest.fixture
def isolated_registry(monkeypatch):
    from rag_langgraph.models import runtime
    registry = runtime.ModelRegistry()
    monkeypatch.setattr(runtime, 'registry', registry)
    return registry


@pytest.fixture
def inference_app():
    from fastapi import FastAPI
    from inference.routers import classify, compress, embed, generate, rerank
    app = FastAPI()
    for module in (classify, compress, embed, generate, rerank):
        app.include_router(module.router, prefix='/inference')
    return app


@pytest.fixture
def query_payload():
    """Explicit model selections for test requests, not an application preset."""
    def build(**overrides):
        payload = {
            'query': '问题', 'collection': 'test_collection',
            'search_method': 'hyde_with_hybrid',
            'embedding_model': 'BAAI/bge-base-en-v1.5', 'embedding_dim': 768,
            'hyde_provider': 'siliconflow', 'hyde_model': 'Qwen/Qwen2.5-7B-Instruct',
            'rerank_method': 'monot5', 'rerank_model': 'castorini/monot5-base-msmarco-10k',
            'compression_method': 'recomp_extractive', 'compression_model': 'fangyuan/nq_extractive_compressor',
            'answer_model': 'Qwen/Qwen2.5-7B-Instruct',
        }
        payload.update(overrides)
        return payload
    return build
