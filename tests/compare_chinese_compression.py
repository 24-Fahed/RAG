"""Real-model evidence collection; no automatic quality verdict or deployment.

Run from the repository root with python -m tests.compare_chinese_compression.
Dependencies: inference/requirements.txt. --answers additionally uses the existing
SiliconFlow credentials file and makes five paid API requests per case.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import math
import gc


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criteria', type=Path, default=Path(__file__).with_name('issue7-quality-criteria.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratio', type=float, default=.8)
    parser.add_argument('--baseline-revision', required=True, help='Reviewed full commit SHA of old LLMLingua checkpoint')
    parser.add_argument('--tokenizer-revision', required=True, help='Full commit SHA of answer-model tokenizer')
    parser.add_argument('--answers', action='store_true')
    parser.add_argument('--answer-max-out-len', type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.ratio <= 1:
        parser.error('--ratio must be in (0, 1]')
    for revision in (args.baseline_revision, args.tokenizer_revision):
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            parser.error('revision arguments must be full lowercase commit SHAs')
    if args.output.exists():
        parser.error('output exists; choose a new evidence filename')

    # Imports happen after argument validation so --help needs no model packages.
    from rag_langgraph.models.semantic_highlight import select_sentences
    from tests.compression_candidates import CANDIDATES, load_candidate

    criteria = json.loads(args.criteria.read_text(encoding='utf-8'))
    answer_model = criteria['comparison_protocol']['answer_model']
    old_model = 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'

    evidence = {
        'status': 'running', 'stage': 'initialization',
        'quality_verdict': 'pending_independent_claim_review',
        'criteria': criteria, 'criteria_sha256': digest(args.criteria.read_text(encoding='utf-8')),
        'configuration': {
            'ratio': args.ratio,
            'candidates': CANDIDATES,
            'candidate_budget': 'Shared answer tokenizer; ceil(source_tokens * ratio); whole sentence overshoot reported',
            'candidate_scoring': 'Mean keep probability per original sentence; stable source-order tie breaking; no threshold tuning',
            'baseline': {'model': old_model, 'revision': args.baseline_revision, 'use_llmlingua2': True},
            'measurement_tokenizer': {'model': answer_model, 'revision': args.tokenizer_revision},
            'answer': {'enabled': args.answers, 'model': answer_model, 'max_tokens': args.answer_max_out_len,
                       'temperature': .2, 'top_p': 'provider default', 'seed': 'not sent',
                       'model_revision': 'not exposed by existing API',
                       'prompt': 'unchanged SiliconFlowGenerator.answer; same across all arms'},
            'dependency_versions': {},
        },
        'limitations': ['Old LLMLingua2 rate is a historical baseline, not the shared candidate token budget.',
                       'Whole sentences cause unequal achieved token budgets; overshoot is measured.',
                       'NAVER uses raw keep probabilities averaged per source sentence, not native process threshold rounding.',
                       'Candidate native token windows differ and are reported.',
                       'Candidate parameter dtypes can differ; single-run timing is descriptive, not a selection criterion.',
                       'Reconstructed cases are not exact original issue request replays.',
                       'Model/API randomness and unavailable server model revision limit reproducibility.',
                       'No keyword-based or model-self-rated quality verdict.'],
        'cases': [],
    }

    def save():
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from rag_langgraph.models.compressors import LLMLinguaCompressor
        from rag_langgraph.models.remote_generator import SiliconFlowGenerator

        evidence['configuration']['dependency_versions'] = {
            name: importlib.metadata.version(name) for name in
            ['torch', 'transformers', 'llmlingua', 'nltk', 'huggingface-hub', 'spacy']
        }
        tokenizer = AutoTokenizer.from_pretrained(answer_model, revision=args.tokenizer_revision)
        generator = SiliconFlowGenerator(answer_model) if args.answers else None
        for case in criteria['cases']:
            source = case['source_context']
            row = {'id': case['id'], 'query': case['query'], 'source_context': source,
                   'query_sha256': digest(case['query']), 'source_sha256': digest(source), 'arms': []}
            evidence['cases'].append(row)
        # Only one compression checkpoint occupies GPU memory at a time.
        for name in ['original_uncompressed', 'old_compressor', *CANDIDATES]:
            evidence['stage'] = 'initialization'
            evidence['active_arm'] = name
            save()
            model = None
            if name == 'old_compressor':
                baseline_path = snapshot_download(repo_id=old_model, revision=args.baseline_revision)
                model = LLMLinguaCompressor(baseline_path)
            elif name in CANDIDATES:
                model = load_candidate(name)
            evidence['stage'] = 'comparison'
            save()
            for case, row in zip(criteria['cases'], evidence['cases']):
                source = case['source_context']
                source_tokens = len(tokenizer.encode(source, add_special_tokens=False))
                target = math.ceil(source_tokens * args.ratio)
                start = time.perf_counter()
                sentence_scores = None
                if name in CANDIDATES:
                    sentences, sentence_scores = model.score_sentences(case['query'], source)
                    compressed = select_sentences(sentences, sentence_scores, source, args.ratio, tokenizer)
                else:
                    compressed = source if model is None else model.compress(case['query'], source, args.ratio)
                compression_seconds = time.perf_counter() - start
                tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
                arm = {'name': name, 'compressed_context': compressed, 'sha256': digest(compressed),
                       'compression_seconds': compression_seconds, 'source_chars': len(source),
                       'output_chars': len(compressed), 'source_tokens': source_tokens, 'output_tokens': tokens,
                       'target_output_tokens': target, 'tokens_above_target': max(0, tokens - target),
                       'target_applied': name in CANDIDATES, 'sentence_scores': sentence_scores,
                       'token_retention': tokens / source_tokens if source_tokens else None,
                       'actually_compressed': tokens < source_tokens, 'answer_status': 'not_requested'}
                row['arms'].append(arm)
                if name in CANDIDATES:
                    parameter = next(model.model.parameters())
                    arm['model_device'] = str(parameter.device)
                    arm['model_parameter_dtype'] = str(parameter.dtype)
                    arm['source_sentences'] = sentences
                    arm['native_model_window'] = (model.model.max_length if name == 'zilliz_bilingual'
                                                  else model.model.max_len)
                save()  # Preserve compression evidence even if answer API fails.
                if generator:
                    start = time.perf_counter()
                    arm['answer'] = generator.answer(case['query'], compressed, args.answer_max_out_len)
                    arm['answer_seconds'] = time.perf_counter() - start
                    arm['answer_status'] = 'returned_not_evaluated'
                    save()
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        evidence['status'] = 'completed_awaiting_independent_review'
        save()
    except Exception as exc:
        evidence['status'] = 'failed'
        # Do not persist exception bodies that may contain provider responses.
        evidence['failure_type'] = type(exc).__name__
        save()
        raise


if __name__ == '__main__':
    main()
