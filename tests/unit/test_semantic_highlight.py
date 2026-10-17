"""Offline contracts; these do not measure real model quality."""
import sys
from types import SimpleNamespace

import pytest

from rag_langgraph.models.semantic_highlight import (
    MODEL_ID, MODEL_REVISION, SemanticHighlightCompressor, split_sentences,
)


def compressor(scores):
    instance = SemanticHighlightCompressor.__new__(SemanticHighlightCompressor)
    seen = {}

    def process(**kwargs):
        seen.update(kwargs)
        sentences = kwargs['sentence_splitter'](kwargs['context'])
        return {'sentence_probabilities': scores,
                'kept_sentences': [s for s, p in zip(sentences, scores) if p > 0],
                'removed_sentences': [s for s, p in zip(sentences, scores) if p == 0]}

    instance.tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
    instance.model = SimpleNamespace(process=process, max_length=8192)
    return instance, seen


@pytest.mark.parametrize('text', [
    '  呼呼呼是提丰粉丝吹丰机。\n哐哐哐是莱伊粉丝洗伊机！  ',
    '不是甲，而是乙；仅适用于版本 3.14。价格为 12.50 元。',
    '\n\n“别名是什么？”\n第二行没有句号  ',
    'English sentence. 中文一句。Final line',
    '一句话\n\n下一段\n',
])
def test_original_spans_are_lossless(text):
    assert ''.join(split_sentences(text)) == text


@pytest.mark.parametrize('text, expected', [
    (
        '来源：萌娘百科社区内容镜像 moegirl.uk《呼呼呼》 '
        'https://moegirl.uk/index.php?title=呼呼呼&variant=zh-sg',
        ['来源：萌娘百科社区内容镜像 moegirl.uk《呼呼呼》 '
         'https://moegirl.uk/index.php?title=呼呼呼&variant=zh-sg'],
    ),
    (
        '链接 https://example.com/path!part?q=what?yes#section。下一句？',
        ['链接 https://example.com/path!part?q=what?yes#section。', '下一句？'],
    ),
    (
        'See HTTPS://example.com/path?q=yes. Is it available? Yes!',
        ['See HTTPS://example.com/path?q=yes. ', 'Is it available? ', 'Yes!'],
    ),
    (
        '链接“https://example.com/path?q=yes”在哪里？\n下一句。',
        ['链接“https://example.com/path?q=yes”在哪里？\n', '下一句。'],
    ),
    ('谁来了? 下一句！谁走了？', ['谁来了? ', '下一句！', '谁走了？']),
    ('Who came? Who left! Final.', ['Who came? ', 'Who left! ', 'Final.']),
])
def test_urls_stay_whole_and_prose_punctuation_still_splits(text, expected):
    assert split_sentences(text) == expected
    assert ''.join(expected) == text


def test_compression_cannot_select_a_url_without_its_query():
    instance, _ = compressor([.1, .9])
    source = '来源：https://moegirl.uk/index.php?title=呼呼呼&variant=zh-sg'
    assert instance.compress('来源是什么？', '背景。' + source, .1) == source


def test_whole_relations_and_original_order():
    instance, seen = compressor([.8, .99, .01])
    text = '呼呼呼是提丰粉丝吹丰机。哐哐哐是莱伊粉丝洗伊机。无关新闻。'
    output = instance.compress('这两个声音分别指谁？', text, .8)
    assert output == '呼呼呼是提丰粉丝吹丰机。哐哐哐是莱伊粉丝洗伊机。'
    assert seen['question'] == '这两个声音分别指谁？'
    assert seen['threshold'] == 0.0  # ratio is not a probability threshold
    assert seen['title'] is None and seen['respect_sentence_boundaries']


def test_budget_keeps_complete_sentence_and_ratio_one_preserves_everything():
    instance, _ = compressor([.1, .9])
    text = '甲不是乙。丙仅在版本三支持丁。'
    assert instance.compress('条件是什么？', text, .1) == '丙仅在版本三支持丁。'
    assert instance.compress('条件是什么？', text, 1) == text


@pytest.mark.parametrize('scores', [[], [.2], [float('nan'), .2], [-.1, .2], ['x', .2], [True, .2]])
def test_invalid_model_metrics_fail(scores):
    instance, _ = compressor(scores)
    # Bypass the mock partition for nonnumeric scores so validation sees the payload.
    instance.model.process = lambda **kw: {'sentence_probabilities': scores}
    with pytest.raises(ValueError):
        instance.compress('问题', '第一句。第二句。')


def test_normalized_or_fragmented_model_text_fails():
    instance, _ = compressor([.9])
    instance.model.process = lambda **kw: {
        'sentence_probabilities': [.9], 'kept_sentences': ['哐 哐 哐。'], 'removed_sentences': [],
    }
    with pytest.raises(ValueError, match='source spans'):
        instance.compress('问题', '哐哐哐。')


@pytest.mark.parametrize('ratio', [0, -1, 1.1, float('nan'), float('inf'), True])
def test_bad_ratio_fails(ratio):
    instance, _ = compressor([.9])
    with pytest.raises(ValueError, match='ratio'):
        instance.compress('问题', '证据。', ratio)


def test_long_context_is_passed_whole_and_long_query_rejected():
    text = ('罕见别名仍是完整原句。' * 1000) + '结尾数字为 2026。'
    instance, seen = compressor([.5] * len(split_sentences(text)))
    assert instance.compress('问题', text, 1) == text
    assert seen['context'] == text
    with pytest.raises(ValueError, match='token window'):
        instance.compress('问' * 8192, text)


def test_oversized_sentence_is_rejected_before_upstream_can_truncate():
    instance, seen = compressor([.9])
    instance.model.max_length = 32
    # With a two-token query and the reserved 16-token margin, 14 remain.
    assert instance.compress('问题', '字' * 13 + '。', .5) == '字' * 13 + '。'
    seen.clear()
    with pytest.raises(ValueError, match='Sentence 1.*partially scored'):
        instance.compress('问题', '字' * 14 + '。', .5)
    assert not seen


@pytest.mark.parametrize('payload', [None, [], 'invalid'])
def test_invalid_model_payload_fails_explicitly(payload):
    instance, _ = compressor([.9])
    instance.model.process = lambda **kw: payload
    with pytest.raises(ValueError, match='metrics mapping'):
        instance.compress('问题', '证据。')


def test_pinned_snapshot_and_nested_tokenizer(monkeypatch):
    seen = {}
    config = SimpleNamespace(base_model_config={'model_type': 'xlm-roberta'})

    def download(**kwargs):
        seen['snapshot'] = kwargs
        return '/cache/pinned-snapshot'

    def load_config(path, **kwargs):
        seen['config'] = (path, kwargs)
        return config

    def load_model(path, **kwargs):
        seen['model'] = (path, kwargs)
        return SimpleNamespace(tokenizer=object())

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=load_config),
        AutoModel=SimpleNamespace(from_pretrained=load_model)))
    SemanticHighlightCompressor(MODEL_ID)
    assert seen['snapshot']['revision'] == MODEL_REVISION
    assert config.tokenizer_name_or_path == '/cache/pinned-snapshot'
    assert seen['model'][1]['local_files_only'] is True
    config.base_model_config = None
    with pytest.raises(ValueError, match='backbone'):
        SemanticHighlightCompressor(MODEL_ID)
    with pytest.raises(ValueError, match='reviewed checkpoint'):
        SemanticHighlightCompressor('another/model')


def test_unselected_candidate_is_not_a_public_adapter(query_payload, inference_app):
    from fastapi.testclient import TestClient
    from server.query_options import QueryRequest
    with pytest.raises(ValueError):
        QueryRequest(**query_payload(compression_method='semantic_highlight', compression_model=MODEL_ID))
    with TestClient(inference_app) as client:
        result = client.post('/inference/compress', json={
            'query': '问题', 'context': '中文证据。', 'method': 'semantic_highlight',
            'model_name': MODEL_ID, 'ratio': .8,
        })
    assert result.status_code == 422
