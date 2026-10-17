"""Evidence must survive initialization failures without storing error bodies."""
import json
import sys
from types import SimpleNamespace

import pytest

from tests import compare_chinese_compression as comparison


def test_initialization_failure_is_recorded_without_sensitive_message(tmp_path, monkeypatch):
    output = tmp_path / 'evidence.json'
    monkeypatch.setattr(sys, 'argv', [
        'comparison', '--output', str(output), '--baseline-revision', 'a' * 40,
        '--tokenizer-revision', 'b' * 40,
    ])

    def fail(*args, **kwargs):
        raise RuntimeError('sensitive provider response')

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=fail))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=fail)))
    monkeypatch.setattr(comparison.importlib.metadata, 'version', lambda name: 'test-only')
    with pytest.raises(RuntimeError):
        comparison.main()
    raw = output.read_text(encoding='utf-8')
    evidence = json.loads(raw)
    assert evidence['status'] == 'failed'
    assert evidence['stage'] == 'initialization'
    assert evidence['failure_type'] == 'RuntimeError'
    assert evidence['cases'] == []
    assert 'sensitive provider response' not in raw


def test_five_arm_compression_only_evidence(tmp_path, monkeypatch):
    from tests import compression_candidates
    from rag_langgraph.models import compressors
    from rag_langgraph.models.semantic_highlight import split_sentences

    output = tmp_path / 'five-arms.json'
    criteria = tmp_path / 'criteria.json'
    criteria.write_text(json.dumps({'comparison_protocol': {'answer_model': 'test/tokenizer'},
                                   'cases': [{'id': 'case', 'query': '问题',
                                              'source_context': '甲乙。丙丁。戊己。'}]}), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['comparison', '--output', str(output), '--criteria', str(criteria),
                                    '--baseline-revision', 'a' * 40, '--tokenizer-revision', 'b' * 40])
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda **kw: '/pinned'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(comparison.importlib.metadata, 'version', lambda name: 'test-only')
    monkeypatch.setattr(compressors, 'LLMLinguaCompressor', lambda path: SimpleNamespace(
        compress=lambda query, source, ratio: source[:6]))
    loaded = []

    def load(name):
        loaded.append(name)
        return SimpleNamespace(model=SimpleNamespace(max_length=512, max_len=8190,
                               parameters=lambda: iter([SimpleNamespace(device='cpu', dtype='test-float')])),
                               score_sentences=lambda q, s: (split_sentences(s), [.9, .8, .1]))

    monkeypatch.setattr(compression_candidates, 'load_candidate', load)
    comparison.main()
    evidence = json.loads(output.read_text(encoding='utf-8'))
    assert evidence['status'] == 'completed_awaiting_independent_review'
    arms = evidence['cases'][0]['arms']
    assert [a['name'] for a in arms] == ['original_uncompressed', 'old_compressor', *compression_candidates.CANDIDATES]
    assert loaded == list(compression_candidates.CANDIDATES)
    assert all(a['answer_status'] == 'not_requested' for a in arms)
    assert all(a['target_output_tokens'] == 8 for a in arms)
    assert all(a['tokens_above_target'] == 1 and a['target_applied'] for a in arms[2:])
