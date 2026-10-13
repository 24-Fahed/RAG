"""
重排序模型封装。

支持的模型：
- MonoT5: castorini/monot5-base-msmarco-10k
- BGE: BAAI/bge-reranker-v2-m3
- RankLLaMA: castorini/rankllama-v1-7b-lora-passage
- TILDE: ielab/TILDEv2-TILDE200-exp
"""

import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """重排序器抽象基类。"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        根据与查询的相关性对文档进行重排序。

        Args:
            query: 用户的问题。
            documents: 文档文本列表。

        Returns:
            包含 'content'、'score'、'metadata' 键的字典列表，按分数降序排列。
        """
        ...


class MonoT5Reranker(BaseReranker):
    """
    使用 T5 进行相关性评分的 MonoT5 重排序器。

    构造输入："Query: ... Document: ... Relevant:"
    T5 输出 "true"/"false"，P(true) 即为相关性分数。
    """

    def __init__(self, model_name: str = "castorini/monot5-base-msmarco-10k"):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded MonoT5 reranker: {model_name}")

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        results = []
        for doc in documents:
            score = self._score_single(query, doc)
            results.append({"content": doc, "score": score, "metadata": {}})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _score_single(self, query: str, document: str) -> float:
        import torch
        input_text = f"Query: {query} Document: {document} Relevant:"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True
            )
            # 获取 "true" 词元的概率
            true_token_id = self.tokenizer.encode("true")[0]
            false_token_id = self.tokenizer.encode("false")[0]
            logits = outputs.scores[0][0]
            true_logit = logits[true_token_id].item()
            false_logit = logits[false_token_id].item()
            # 通过 Softmax 获取概率
            import math
            true_prob = math.exp(true_logit) / (math.exp(true_logit) + math.exp(false_logit))
            return true_prob


class BGEReranker(BaseReranker):
    """BGE 交叉编码器重排序器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded BGE reranker: {model_name}")

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        import torch
        pairs = [[query, doc] for doc in documents]
        features = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)

        scores = scores.tolist()
        if isinstance(scores, float):
            scores = [scores]

        results = [
            {"content": doc, "score": score, "metadata": {}}
            for doc, score in zip(documents, scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class TILDEReranker(BaseReranker):
    """使用查询扩展的 TILDE 重排序器。"""

    def __init__(self, model_name: str = "ielab/TILDEv2-TILDE200-exp"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded TILDE reranker: {model_name}")

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        # TILDE 使用文档扩展 + 词项评分
        import torch
        results = []
        for doc in documents:
            inputs = self.tokenizer(
                query, doc, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)
            with torch.no_grad():
                score = self.model(**inputs).logits.squeeze(-1).item()
            results.append({"content": doc, "score": score, "metadata": {}})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class RankLLaMAReranker(BaseReranker):
    """使用 LoRA 的 RankLLaMA 重排序器。"""

    def __init__(self, model_name: str = "castorini/rankllama-v1-7b-lora-passage"):
        try:
            from peft import PeftModel
            from transformers import LlamaForCausalLM, LlamaTokenizer
        except ImportError:
            raise ImportError("RankLLaMA requires peft: pip install peft")

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model_name = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(base_model, model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded RankLLaMA reranker: {model_name}")

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        import torch
        results = []
        for doc in documents:
            prompt = f"Query: {query} Document: {doc}\nRelevant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True
                )
                yes_token_id = self.tokenizer.encode("Yes")[0]
                logits = outputs.scores[0][0]
                score = torch.sigmoid(logits[yes_token_id]).item()
            results.append({"content": doc, "score": score, "metadata": {}})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


RERANKER_REGISTRY = {
    "monot5": MonoT5Reranker,
    "bge": BGEReranker,
    "tilde": TILDEReranker,
    "rankllama": RankLLaMAReranker,
}


def get_reranker(model_name: str = "monot5", checkpoint: str | None = None) -> BaseReranker:
    from rag_langgraph.models.runtime import registry
    from rag_langgraph.model_options import RERANK_MODELS
    cls = RERANKER_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown reranker adapter: {model_name}")
    checkpoint = checkpoint or RERANK_MODELS[model_name]
    return registry.get(("reranker", model_name, checkpoint), lambda: cls(checkpoint))
