"""Pinned NAVER sentence scores shared by production and comparison experiments."""
import math
import os
import string

from rag_langgraph.models.semantic_highlight import select_sentences, split_sentences

MODEL_ID = 'naver/xprovence-reranker-bgem3-v1'
MODEL_REVISION = '8dd5d79b6b57b3498a1180352daaf8c457af832b'
BUDGET_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
BUDGET_MODEL_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
XPROVENCE_CHECKPOINTS = {
    'naver_xprovence_v1': {'model': MODEL_ID, 'revision': MODEL_REVISION},
    'naver_xprovence_v2': {
        'model': 'naver/xprovence-reranker-bgem3-v2',
        'revision': 'f2798d357213c40164bdd79557d6d2e43ca7d8d2',
    },
}


def build_blocks(tokenizer, query, context, max_length):
    """Match NAVER query normalization/input format, preserving source spans."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must contain non-whitespace text')
    if not isinstance(context, str):
        raise TypeError('context must be a string')
    sentences = split_sentences(context)
    if not context.strip():
        return [], []
    if ''.join(sentences) != context:
        raise ValueError('Sentence splitting did not preserve the input')
    normalized = ' '.join(''.join(c for c in query.lower() if c not in string.punctuation).split())
    query_ids = tokenizer.encode(normalized, add_special_tokens=False)
    specials = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id]
    if any(not isinstance(x, int) for x in specials):
        raise ValueError('Checkpoint requires CLS, SEP and EOS token IDs')
    if not query_ids or len(query_ids) + 3 >= max_length:
        raise ValueError('Query leaves insufficient room in the model token window')
    prefix = [specials[0]] + query_ids + [specials[1]]
    capacity = max_length - len(prefix) - 1
    blocks, ids, ranges = [], list(prefix), []
    for index, sentence in enumerate(sentences):
        # Match upstream per-sentence tokenization (space before later sentences),
        # retaining original text solely for output assembly.
        tokens = tokenizer.encode((' ' if index else '') + sentence.strip(), add_special_tokens=False)
        if not tokens or len(tokens) > capacity:
            raise ValueError(f'Sentence {index + 1} has no tokens or exceeds the model token window')
        if len(ids) + len(tokens) + 1 > max_length:
            blocks.append((ids + [specials[2]], ranges))
            ids, ranges = list(prefix), []
        ranges.append((index, len(ids), len(ids) + len(tokens)))
        ids.extend(tokens)
    if ranges:
        blocks.append((ids + [specials[2]], ranges))
    return sentences, blocks


class XProvenceScorer:
    def __init__(self, name):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        import torch

        if name not in ('naver_xprovence_v1', 'naver_xprovence_v2'):
            raise ValueError('Unknown reviewed NAVER checkpoint')
        spec = XPROVENCE_CHECKPOINTS[name]
        snapshot = snapshot_download(repo_id=spec['model'], revision=spec['revision'],
                                     allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt', '*.model'],
                                     endpoint=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
        # The nested tokenizer uses config._name_or_path, now the pinned local snapshot.
        self.model = AutoModel.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        self.tokenizer = self.model.tokenizer

    def score_sentences(self, query, context):
        import torch

        sentences, blocks = build_blocks(self.tokenizer, query, context, self.model.max_len)
        scores = [None] * len(sentences)
        with torch.inference_mode():
            for ids, ranges in blocks:
                tensor = torch.tensor([ids], dtype=torch.long, device=self.model.device)
                result = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor))
                logits = result.compression_logits
                if tuple(logits.shape) != (1, len(ids), 2):
                    raise ValueError('Unexpected compression logits shape')
                probabilities = torch.softmax(logits.float(), dim=-1)[0, :, 1]
                if not torch.isfinite(probabilities).all().item():
                    raise ValueError('Nonfinite compression probabilities')
                for index, start, end in ranges:
                    if scores[index] is not None:
                        raise ValueError('Duplicate sentence score')
                    scores[index] = probabilities[start:end].mean().item()
        if any(score is None for score in scores):
            raise ValueError('Not every original sentence was scored')
        return sentences, scores


class XProvenceCompressor(XProvenceScorer):
    """V1 only; rank original sentences against the comparison's Qwen budget.

    The last complete sentence may overshoot the target. Small-sample relative
    performance does not guarantee preservation of every fact or relation.
    """

    def __init__(self, model_name: str = MODEL_ID):
        if model_name != MODEL_ID:
            raise ValueError(f'xprovence requires the reviewed checkpoint {MODEL_ID}')
        super().__init__('naver_xprovence_v1')
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        # Only tokenizer assets are fetched, never answer-model weights.
        snapshot = snapshot_download(
            repo_id=BUDGET_MODEL_ID, revision=BUDGET_MODEL_REVISION,
            allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken'],
            endpoint=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'),
        )
        self.budget_tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
        )

    def compress(self, query: str, context: str, ratio: float = 0.8) -> str:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError('ratio must be finite and in (0, 1]')
        sentences, scores = self.score_sentences(query, context)
        if not sentences:
            return context
        return select_sentences(sentences, scores, context, ratio, self.budget_tokenizer)
