import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.answer_chinese_compression_remote import ARMS, MODEL, digest, run, validate


def fixture_data():
    criteria = json.loads((Path(__file__).parents[1] / 'issue7-quality-criteria.json').read_text(encoding='utf-8'))
    evidence = {'status': 'completed_awaiting_independent_review',
                'configuration': {'measurement_tokenizer': {'model': MODEL}}, 'cases': []}
    for c in criteria['cases']:
        evidence['cases'].append(dict(id=c['id'], query=c['query'], source_context=c['source_context'],
            query_sha256=digest(c['query']), source_sha256=digest(c['source_context']),
            arms=[dict(name=n, compressed_context=c['source_context'], sha256=digest(c['source_context']),
                       answer_status='not_requested') for n in sorted(ARMS)]))
    return evidence, criteria


def test_twenty_calls_with_write_ahead_and_fixed_settings(tmp_path):
    evidence, criteria = fixture_data()
    validate(evidence, criteria)
    output = tmp_path / 'answers.json'
    calls = []

    def answer(query, context, limit):
        saved = json.loads(output.read_text(encoding='utf-8'))
        assert saved['remote_attempt_count'] == len(calls) + 1
        assert sum(a['answer_status'] == 'in_flight_or_unknown' for c in saved['cases'] for a in c['arms']) == 1
        assert limit == 1024
        calls.append(query)
        return 'test answer'

    assert run(evidence, output, SimpleNamespace(answer=answer))
    assert len(calls) == 20
    assert evidence['remote_attempt_count'] == 20


def test_failure_consumes_attempt_and_stops_without_retry_or_error_body(tmp_path):
    evidence, _ = fixture_data()
    output = tmp_path / 'failed.json'
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError('sensitive-response-secret')

    assert not run(evidence, output, SimpleNamespace(answer=fail))
    raw = output.read_text(encoding='utf-8')
    assert calls == [1] and evidence['remote_attempt_count'] == 1
    assert 'sensitive-response-secret' not in raw
    assert evidence['status'] == 'failed_requires_manual_review'
    assert sum(a['answer_status'] == 'not_requested' for c in evidence['cases'] for a in c['arms']) == 19


def test_input_mutations_and_previous_answers_are_rejected():
    evidence, criteria = fixture_data()
    evidence['cases'][0]['query'] += 'changed'
    with pytest.raises(ValueError):
        validate(evidence, criteria)
    evidence, criteria = fixture_data()
    evidence['cases'][0]['arms'][0]['answer'] = 'existing'
    with pytest.raises(ValueError):
        validate(evidence, criteria)
