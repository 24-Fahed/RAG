"""Append local deterministic answers to five-arm compression evidence in a NEW file."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def answer_messages(query, context):
    # Reuse the exact production template without calling its HTTP transport.
    from rag_langgraph.models.remote_generator import SiliconFlowGenerator

    class TemplateOnly(SiliconFlowGenerator):
        def _chat(self, system, user, max_out_len):
            return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

    return TemplateOnly().answer(query, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compression-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new evidence filename')
    if len(args.model_revision) != 40 or any(c not in '0123456789abcdef' for c in args.model_revision):
        parser.error('model revision must be a full lowercase commit SHA')
    if args.max_new_tokens <= 0:
        parser.error('max-new-tokens must be positive')
    raw = args.compression_evidence.read_bytes()
    evidence = json.loads(raw)
    if evidence['status'] != 'completed_awaiting_independent_review':
        parser.error('compression evidence must have completed all arms')
    expected = {'original_uncompressed', 'old_compressor', 'zilliz_bilingual',
                'naver_xprovence_v1', 'naver_xprovence_v2'}
    if not evidence['cases'] or any({a['name'] for a in c['arms']} != expected or len(c['arms']) != 5
                                    for c in evidence['cases']):
        parser.error('each case must contain exactly the five planned arms')
    if any(a.get('answer_status') != 'not_requested' for c in evidence['cases'] for a in c['arms']):
        parser.error('input must be compression-only evidence')
    model_id = evidence['configuration']['measurement_tokenizer']['model']
    if model_id != 'Qwen/Qwen2.5-7B-Instruct':
        parser.error('this protocol requires Qwen/Qwen2.5-7B-Instruct')
    if args.model_revision != evidence['configuration']['measurement_tokenizer']['revision']:
        parser.error('generation checkpoint and measurement tokenizer must use the same revision')
    evidence['compression_evidence_sha256'] = hashlib.sha256(raw).hexdigest()
    evidence['status'] = 'running'
    evidence['stage'] = 'local_answer_initialization'
    evidence['configuration']['answer'] = {
        'enabled': True, 'transport': 'local_only', 'model': model_id,
        'revision': args.model_revision, 'dtype': 'float16', 'do_sample': False,
        'max_new_tokens': args.max_new_tokens, 'prompt': 'unchanged SiliconFlowGenerator.answer template',
    }

    def save():
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError('Local FP16 answer protocol requires CUDA')
        snapshot = snapshot_download(repo_id=model_id, revision=args.model_revision)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                                   torch_dtype=torch.float16).to('cuda').eval()
        evidence['stage'] = 'local_answer_generation'
        for case in evidence['cases']:
            for arm in case['arms']:
                messages = answer_messages(case['query'], arm['compressed_context'])
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                input_count = inputs['input_ids'].shape[-1]
                if input_count + args.max_new_tokens > model.config.max_position_embeddings:
                    raise ValueError('Answer input and output budget exceed the model window')
                start = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
                generated = output[0, input_count:].tolist()
                eos = model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                stop = ('eos' if generated and generated[-1] in eos_ids else
                        'max_new_tokens' if len(generated) >= args.max_new_tokens else 'other')
                arm.update(answer=tokenizer.decode(generated, skip_special_tokens=True),
                           answer_seconds=time.perf_counter() - start,
                           answer_status='returned_not_evaluated', answer_messages=messages,
                           answer_prompt=prompt, answer_input_token_ids=inputs['input_ids'][0].tolist(),
                           answer_output_token_ids=generated, answer_input_tokens=input_count,
                           answer_output_tokens=len(generated), answer_stop_reason=stop)
                save()
        evidence['status'] = 'completed_awaiting_independent_review'
        save()
    except Exception as exc:
        evidence['status'] = 'failed'
        evidence['failure_type'] = type(exc).__name__
        save()
        raise


if __name__ == '__main__':
    main()
