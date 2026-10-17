"""Offline production contracts; real model quality is evaluated separately."""
import sys
from types import SimpleNamespace

import pytest

from rag_langgraph.models import xprovence
from rag_langgraph.models.semantic_highlight import select_sentences, split_sentences


def test_production_uses_the_comparison_budget_and_original_order():
    instance = xprovence.XProvenceCompressor.__new__(xprovence.XProvenceCompressor)
    text = '甲乙。丙丁。戊己。庚辛。壬癸。'
    sentences, scores = split_sentences(text), [.9, .1, .8, .7, .6]
    instance.score_sentences = lambda query, context: (sentences, scores)
    instance.budget_tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
    # The native scoring tokenizer must not control the token budget.
    instance.tokenizer = SimpleNamespace(encode=lambda *a, **kw: pytest.fail('native budget'))
    expected = select_sentences(sentences, scores, text, .8, instance.budget_tokenizer)
    assert instance.compress('问题', text, .8) == expected == '甲乙。戊己。庚辛。壬癸。'
    assert instance.compress('问题', text, 1) == text


@pytest.mark.parametrize('ratio', [0, 1.1, float('nan'), True])
def test_invalid_ratio_fails_before_model_scoring(ratio):
    instance = xprovence.XProvenceCompressor.__new__(xprovence.XProvenceCompressor)
    instance.score_sentences = lambda *a: pytest.fail('invalid request reached model')
    with pytest.raises(ValueError, match='ratio'):
        instance.compress('问题', '证据。', ratio)


def test_fixed_v1_and_tokenizer_only_download(monkeypatch):
    seen = {}
    monkeypatch.delenv('HF_ENDPOINT', raising=False)
    monkeypatch.setattr(xprovence.XProvenceScorer, '__init__',
                        lambda self, name: seen.update(candidate=name))

    def download(**kwargs):
        seen['download'] = kwargs
        return '/cache/qwen-tokenizer'

    def tokenizer(path, **kwargs):
        seen['tokenizer'] = path, kwargs
        return object()

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    xprovence.XProvenceCompressor()
    assert seen['candidate'] == 'naver_xprovence_v1'
    spec = seen['download']
    assert spec['repo_id'] == 'Qwen/Qwen2.5-7B-Instruct'
    assert spec['revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28'
    assert spec['endpoint'] == 'https://hf-mirror.com'
    assert not any('safetensors' in item or '*.bin' == item for item in spec['allow_patterns'])
    assert seen['tokenizer'] == ('/cache/qwen-tokenizer', {
        'local_files_only': True, 'trust_remote_code': False})
    with pytest.raises(ValueError, match='reviewed checkpoint'):
        xprovence.XProvenceCompressor('naver/xprovence-reranker-bgem3-v2')


def test_production_and_experiments_share_the_pinned_scorer(monkeypatch):
    from tests import compression_candidates
    seen = {}

    class Model:
        tokenizer = object()

        def to(self, device):
            seen['device'] = device
            return self

        def eval(self):
            return self

    def download(**kwargs):
        seen['download'] = kwargs
        return '/cache/naver-v1'

    def load(path, **kwargs):
        seen['model'] = path, kwargs
        return Model()

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert compression_candidates.XProvenceScorer is xprovence.XProvenceScorer
    assert compression_candidates.build_blocks is xprovence.build_blocks
    compression_candidates.load_candidate('naver_xprovence_v1')
    assert seen['download']['revision'] == '8dd5d79b6b57b3498a1180352daaf8c457af832b'
    assert seen['model'] == ('/cache/naver-v1', {'trust_remote_code': True, 'local_files_only': True})
    compression_candidates.load_candidate('naver_xprovence_v2')
    assert seen['download']['revision'] == 'f2798d357213c40164bdd79557d6d2e43ca7d8d2'


def test_selected_adapter_is_available_through_public_and_worker_apis(query_payload, inference_app, monkeypatch):
    from fastapi.testclient import TestClient
    from server.query_options import QueryRequest
    from inference.routers import compress
    from rag_langgraph.models.compressors import COMPRESSOR_REGISTRY
    from rag_langgraph.model_options import COMPRESSION_MODELS
    from server.api_examples import QUERY_EXAMPLES

    assert COMPRESSOR_REGISTRY['xprovence'] is xprovence.XProvenceCompressor
    assert 'semantic_highlight' not in COMPRESSOR_REGISTRY
    assert COMPRESSION_MODELS['xprovence'] == xprovence.MODEL_ID
    example = QUERY_EXAMPLES['explicit_models']['value']
    assert (example['compression_method'], example['compression_model'], example['compression_ratio']) == (
        'xprovence', xprovence.MODEL_ID, .8)
    request = QueryRequest(**query_payload(compression_method='xprovence', compression_model=xprovence.MODEL_ID,
                                           compression_ratio=.8))
    assert request.pipeline_kwargs()['compression_model'] == xprovence.MODEL_ID
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(compress=lambda query, context, ratio: context)

    monkeypatch.setattr(compress, 'get_compressor', factory)
    with TestClient(inference_app) as client:
        response = client.post('/inference/compress', json={
            'query': '问题', 'context': '中文证据。', 'method': 'xprovence',
            'model_name': xprovence.MODEL_ID, 'ratio': .8,
        })
    assert response.status_code == 200 and response.json()['compressed'] == '中文证据。'
    assert seen == {'method': 'xprovence', 'model_name': xprovence.MODEL_ID}
