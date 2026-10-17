"""
压缩模型封装。

支持的模型：
- recomp_extractive: fangyuan/nq_extractive_compressor（基于 Contriever）
- recomp_abstractive: fangyuan/nq_abstractive_compressor（基于 T5）
- xprovence: naver/xprovence-reranker-bgem3-v1
- llmlingua: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
"""

import logging
from abc import ABC, abstractmethod
from rag_langgraph.models.xprovence import XProvenceCompressor


logger = logging.getLogger(__name__)


class BaseCompressor(ABC):
    """上下文压缩器抽象基类。"""

    @abstractmethod
    def compress(self, query: str, context: str, ratio: float = 0.6) -> str:
        """
        在保留与查询相关信息的同时压缩上下文。

        Args:
            query: 用户的问题。
            context: 要压缩的输入上下文。
            ratio: 目标压缩比率（保留原始内容的比例）。

        Returns:
            压缩后的上下文字符串。
        """
        ...


class RecompExtractiveCompressor(BaseCompressor):
    """
    使用 Contriever 的抽取式压缩。

    将上下文分割为句子，对每个句子与查询进行评分，
    按相似度选择 top_n 个句子，并按原始顺序重新组装。
    """

    def __init__(self, model_name: str = "fangyuan/nq_extractive_compressor"):
        from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded Recomp extractive compressor: {model_name}")

    def compress(self, query: str, context: str, ratio: float = 0.6) -> str:
        sentences = self._split_sentences(context)
        if not sentences:
            return context

        # 对每个句子与查询进行评分
        scores = self._score_sentences(query, sentences)

        # 选择 top_n 个句子
        n_keep = max(1, int(len(sentences) * ratio))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_keep]

        # 按原始顺序重新组装
        top_indices.sort()
        selected = [sentences[i] for i in top_indices]
        return " ".join(selected)

    def _split_sentences(self, text: str) -> list[str]:
        """将文本分割为句子。"""
        import re
        sentences = re.split(r'(?<=[。！？])\s*|(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(self, query: str, sentences: list[str]) -> list[float]:
        """使用嵌入模型计算查询与每个句子之间的相似度。"""
        import torch
        scores = []
        query_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            query_emb = self.model(**query_inputs).last_hidden_state.mean(dim=1)

        for sent in sentences:
            sent_inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                sent_emb = self.model(**sent_inputs).last_hidden_state.mean(dim=1)
            similarity = torch.cosine_similarity(query_emb, sent_emb).item()
            scores.append(similarity)

        return scores


class RecompAbstractiveCompressor(BaseCompressor):
    """
    使用 T5 的生成式压缩。

    生成与查询相关的上下文摘要。
    """

    def __init__(self, model_name: str = "fangyuan/nq_abstractive_compressor"):
        from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded Recomp abstractive compressor: {model_name}")

    def compress(self, query: str, context: str, ratio: float = 0.6) -> str:
        import torch
        max_length = max(10, int(len(context.split()) * ratio))
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class LLMLinguaCompressor(BaseCompressor):
    """
    基于 LLMLingua 的提示压缩。

    使用小模型识别并移除不重要的词元。
    """

    def __init__(self, model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"):
        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError("LLMLingua requires: pip install llmlingua")

        self.compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True)
        logger.info(f"Loaded LLMLingua compressor: {model_name}")

    def compress(self, query: str, context: str, ratio: float = 0.6) -> str:
        result = self.compressor.compress_prompt(context, rate=ratio)
        return result["compressed_prompt"]


COMPRESSOR_REGISTRY = {
    "recomp_extractive": RecompExtractiveCompressor,
    "recomp_abstractive": RecompAbstractiveCompressor,
    "llmlingua": LLMLinguaCompressor,
    "xprovence": XProvenceCompressor,
}


def get_compressor(method: str = "recomp_extractive", model_name: str | None = None) -> BaseCompressor:
    from rag_langgraph.models.runtime import registry
    from rag_langgraph.model_options import COMPRESSION_MODELS
    cls = COMPRESSOR_REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Unknown compressor adapter: {method}")
    model_name = model_name or COMPRESSION_MODELS[method]
    return registry.get(("compressor", method, model_name), lambda: cls(model_name))
