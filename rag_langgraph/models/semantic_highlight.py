"""Query-aware, whole-sentence compression using a pinned bilingual checkpoint."""
import math
import re

MODEL_ID = "zilliz/semantic-highlight-bilingual-v1"
MODEL_REVISION = "6dfd9cbee6d9309201b4ff4b4bdd814e1c064491"


def split_sentences(text: str) -> list[str]:
    """Keep exact source spans, including whitespace, punctuation and closing quotes.

    Chinese terminal punctuation and newlines delimit sentences. English periods
    delimit only before whitespace, so decimals and dotted identifiers stay intact.
    Semicolons/commas stay within a sentence to preserve local qualifications.
    HTTP(S) URLs are consumed as source spans before looking for sentence ends.
    """
    # Quotes, whitespace and Chinese terminal punctuation end a bare URL. Leave
    # a trailing period outside it so English prose can still end after a link.
    url = r'https?://[^\s<>"\'“”‘’「」『』。！？]+(?<!\.)'
    ends = re.compile(
        rf'(?P<url>{url})|[。！？!?]+[”’」』"\']*\s*|\.(?=\s)\s+|\n+',
        re.IGNORECASE,
    )
    spans, start = [], 0
    for match in ends.finditer(text):
        if match.group('url') is not None:
            continue
        end = match.end()
        if text[start:end].strip():
            spans.append(text[start:end])
            start = end
    if start < len(text):
        if text[start:].strip():
            spans.append(text[start:])
        elif spans:
            spans[-1] += text[start:]
    return spans


class SemanticHighlightCompressor:
    """Select scored original sentences until the target token budget is reached.

    ratio is a retention target, never the model's probability threshold. The
    final whole sentence can overshoot the target; ratio=1 preserves every span.
    This does not guarantee retention of all facts or cross-sentence relations.
    """

    def __init__(self, model_name: str = MODEL_ID):
        if model_name != MODEL_ID:
            raise ValueError(f"semantic_highlight requires the reviewed checkpoint {MODEL_ID}")
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModel

        snapshot = snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model"],
        )
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        if not getattr(config, "base_model_config", None):
            raise ValueError("Pinned checkpoint must contain its backbone configuration")
        # Upstream initializes its own tokenizer without forwarding revision.
        # A local snapshot keeps that nested load on the same reviewed revision.
        config.tokenizer_name_or_path = snapshot
        self.model = AutoModel.from_pretrained(
            snapshot, config=config, trust_remote_code=True, local_files_only=True,
        )
        self.tokenizer = self.model.tokenizer

    def compress(self, query: str, context: str, ratio: float = 0.6) -> str:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("ratio must be finite and in (0, 1]")
        sentences, scores = self.score_sentences(query, context)
        if not sentences:
            return context
        return select_sentences(sentences, scores, context, ratio, self.tokenizer)

    def score_sentences(self, query: str, context: str) -> tuple[list[str], list[float]]:
        """Expose validated source spans and scores for controlled experiments."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must contain non-whitespace text")
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not context.strip():
            return [], []
        sentences = split_sentences(context)
        if "".join(sentences) != context:
            raise ValueError("Sentence splitting did not preserve the input")
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        if len(query_tokens) >= self.model.max_length - 16:
            raise ValueError("Query leaves insufficient room in the model token window")
        # The pinned upstream fragments before accounting for the query, then
        # truncates oversized fragments when packing blocks. Its returned sentence
        # texts still contain the unscored tail, so text matching cannot detect this.
        # Reserve the same conservative special-token margin as the query guard.
        sentence_capacity = self.model.max_length - len(query_tokens) - 16
        for index, sentence in enumerate(sentences):
            if len(self.tokenizer.encode(sentence, add_special_tokens=False)) > sentence_capacity:
                raise ValueError(
                    f"Sentence {index + 1} exceeds the model token window after the query; "
                    "refusing partially scored compression"
                )
        result = self.model.process(
            question=query, context=context, title=None, threshold=0.0,
            sentence_splitter=split_sentences, strip_sentences=False,
            respect_sentence_boundaries=True, return_sentence_metrics=True,
            return_sentence_texts=True, reorder=False, show_progress=False,
            preprocess_workers=0,
        )
        if not isinstance(result, dict):
            raise ValueError("Model must return a sentence metrics mapping")
        scores = result.get("sentence_probabilities")
        if not isinstance(scores, list) or len(scores) != len(sentences):
            raise ValueError("Model must return one probability per original sentence")
        if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Model returned an invalid sentence probability")
        # At threshold=0 the upstream partition must match the exact source spans.
        # Reject tokenizer-normalized or fragmented results instead of guessing a mapping.
        if (result.get("kept_sentences") != [s for s, p in zip(sentences, scores) if p > 0]
                or result.get("removed_sentences") != [s for s, p in zip(sentences, scores) if p == 0]):
            raise ValueError("Model sentence texts do not match the original source spans")
        return sentences, scores


def select_sentences(sentences, scores, context, ratio, tokenizer):
    """Rank whole source sentences against one explicitly supplied tokenizer."""
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be finite and in (0, 1]")
    if ''.join(sentences) != context or len(scores) != len(sentences):
        raise ValueError("Sentence scores must cover every original span")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Invalid sentence probability")
    budget = math.ceil(len(tokenizer.encode(context, add_special_tokens=False)) * ratio)
    selected = set()
    output = ""
    for index in sorted(range(len(sentences)), key=lambda i: (-scores[i], i)):
        selected.add(index)
        output = "".join(sentence for i, sentence in enumerate(sentences) if i in selected)
        if ratio < 1 and len(tokenizer.encode(output, add_special_tokens=False)) >= budget:
            break
    return output
