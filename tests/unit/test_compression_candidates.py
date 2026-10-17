from types import SimpleNamespace

import pytest

from tests.compression_candidates import CANDIDATES, build_blocks
from rag_langgraph.models.semantic_highlight import select_sentences, split_sentences


def tokenizer():
    return SimpleNamespace(cls_token_id=1, sep_token_id=2, eos_token_id=3,
                           encode=lambda text, **kwargs: [ord(c) for c in text])


def test_all_source_sentences_covered_once_across_blocks():
    text = '第一句。第二句。结尾42。'
    sentences, blocks = build_blocks(tokenizer(), '问?', text, 13)
    assert ''.join(sentences) == text
    assert len(blocks) > 1
    assert [index for _, ranges in blocks for index, _, _ in ranges] == [0, 1, 2]
    for ids, ranges in blocks:
        assert len(ids) <= 13
        assert ids[:3] == [1, ord('问'), 2]  # upstream query punctuation normalization
        assert ids[-1] == 3
        for index, start, end in ranges:
            expected = (' ' if index else '') + sentences[index].strip()
            assert ids[start:end] == [ord(c) for c in expected]


def test_overlong_sentence_or_query_fails_without_truncation():
    with pytest.raises(ValueError, match='Sentence 1'):
        build_blocks(tokenizer(), '问题', '长' * 20, 15)
    with pytest.raises(ValueError, match='Query'):
        build_blocks(tokenizer(), '问' * 20, '短。', 15)


def test_shared_measurement_tokenizer_controls_budget_and_source_order():
    text = '甲乙。丙丁。戊己。'
    sentences = split_sentences(text)
    assert select_sentences(sentences, [.8, .1, .9], text, .5, tokenizer()) == '甲乙。戊己。'
    assert select_sentences(sentences, [.8, .1, .9], text, 1, tokenizer()) == text
    with pytest.raises(ValueError):
        select_sentences(sentences, [.8], text, .8, tokenizer())


def test_three_distinct_pinned_checkpoints():
    assert len(CANDIDATES) == 3
    assert len({x['model'] for x in CANDIDATES.values()}) == 3
    assert all(len(x['revision']) == 40 for x in CANDIDATES.values())
