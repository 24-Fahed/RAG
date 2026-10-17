"""One authorized four-case/five-arm cloud answer run; no resume or retries."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from tests.answer_chinese_compression_local import answer_messages

ARMS = {'original_uncompressed', 'old_compressor', 'zilliz_bilingual',
        'naver_xprovence_v1', 'naver_xprovence_v2'}
MODEL = 'Qwen/Qwen2.5-7B-Instruct'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate(evidence, criteria):
    if evidence.get('status') != 'completed_awaiting_independent_review':
        raise ValueError('Compression must be complete')
    expected = criteria['cases']
    cases = evidence['cases']
    if len(expected) != 4 or len(cases) != 4 or len({c['id'] for c in cases}) != 4:
        raise ValueError('Exactly four distinct authorized cases required')
    if evidence['configuration']['measurement_tokenizer']['model'] != MODEL:
        raise ValueError('Unexpected comparison model')
    by_id = {c['id']: c for c in expected}
    if {c['id'] for c in cases} != set(by_id):
        raise ValueError('Case IDs differ from criteria')
    for case in cases:
        source = by_id[case['id']]
        if case['query'] != source['query'] or case['source_context'] != source['source_context']:
            raise ValueError('Case input differs from authorized criteria')
        if case.get('source_sha256') != digest(case['source_context']) or case.get('query_sha256') != digest(case['query']):
            raise ValueError('Case input hash mismatch')
        arms = case['arms']
        if len(arms) != 5 or {a['name'] for a in arms} != ARMS:
            raise ValueError('Exactly five planned arms required per case')
        for arm in arms:
            if arm.get('answer_status') != 'not_requested' or any(k.startswith('answer') and k != 'answer_status' for k in arm):
                raise ValueError('Input already contains answer data')
            if arm.get('sha256') != digest(arm['compressed_context']):
                raise ValueError('Compressed input hash mismatch')
            if arm['name'] == 'original_uncompressed' and arm['compressed_context'] != case['source_context']:
                raise ValueError('Original arm differs from source')


def run(evidence, output, generator):
    evidence['status'] = 'running'
    evidence['stage'] = 'authorized_remote_answers'
    evidence['remote_attempt_count'] = 0
    evidence['remote_attempt_limit'] = 20
    evidence['configuration']['answer'] = {
        'enabled': True, 'transport': 'https://api.siliconflow.cn/v1/chat/completions',
        'model': MODEL, 'max_tokens': 1024, 'temperature': .2,
        'seed': 'not sent', 'top_p': 'provider default', 'provider_revision': 'not exposed',
        'deterministic': False, 'automatic_retries': 0,
        'prompt': 'unchanged SiliconFlowGenerator.answer template',
    }
    evidence['interruption_policy'] = ('No automatic resume or resend. in_flight_or_unknown means a request '
                                       'may have reached the provider; manually reconcile before any further action.')

    def save():
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    for case in evidence['cases']:
        for arm in case['arms']:
            if evidence['remote_attempt_count'] >= 20:
                raise RuntimeError('Authorized attempt limit reached')
            evidence['remote_attempt_count'] += 1
            arm['answer_attempt_number'] = evidence['remote_attempt_count']
            arm['answer_status'] = 'in_flight_or_unknown'
            arm['answer_messages'] = answer_messages(case['query'], arm['compressed_context'])
            save()  # Write-ahead record: failed/uncertain attempts consume the budget too.
            start = time.perf_counter()
            try:
                arm['answer'] = generator.answer(case['query'], arm['compressed_context'], 1024)
                arm['answer_status'] = 'returned_not_evaluated'
                arm['answer_seconds'] = time.perf_counter() - start
                save()
            except (Exception, KeyboardInterrupt) as exc:
                arm['answer_status'] = 'failed_or_interrupted_delivery_unknown'
                arm['answer_failure_type'] = type(exc).__name__
                evidence['status'] = 'failed_requires_manual_review'
                save()
                return False
    evidence['status'] = 'completed_awaiting_independent_review'
    save()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compression-evidence', type=Path, required=True)
    parser.add_argument('--criteria', type=Path, default=Path(__file__).with_name('issue7-quality-criteria.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.compression_evidence.read_bytes()
    evidence = json.loads(raw)
    criteria_raw = args.criteria.read_text(encoding='utf-8')
    validate(evidence, json.loads(criteria_raw))
    evidence['compression_evidence_sha256'] = hashlib.sha256(raw).hexdigest()
    evidence['answer_authorization_criteria_sha256'] = digest(criteria_raw)
    # Exclusive creation prevents overwriting any previous run, including interrupted runs.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    from rag_langgraph.models.remote_generator import SiliconFlowGenerator
    if not run(evidence, args.output, SiliconFlowGenerator(MODEL)):
        raise SystemExit('Answer run stopped; inspect saved attempt states before proceeding. No request retried.')


if __name__ == '__main__':
    main()
