"""
基于 BERT 的检索分类模型。

判断查询是否需要外部知识检索。
模型：google-bert/bert-base-multilingual-cased（微调）
"""

import logging
from typing import Optional


logger = logging.getLogger(__name__)



class RetrievalClassifier:
    """用于判断检索必要性的 BERT 二分类器。"""

    def __init__(self, model_name: str, weights_path: Optional[str] = None):
        from transformers import BertForSequenceClassification, BertTokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded classification weights from {weights_path}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, query: str) -> int:
        """
        预测查询是否需要检索。

        Args:
            query: 用户的问题。

        Returns:
            0 = 不需要检索，1 = 需要检索。
        """
        import torch
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction

    def batch_predict(self, queries: list[str]) -> list[int]:
        """对一批查询进行预测。"""
        return [self.predict(q) for q in queries]


def get_classifier(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    weights_path: Optional[str] = None,
) -> RetrievalClassifier:
    """按模型与微调权重获取缓存实例。"""
    from rag_langgraph.models.runtime import registry
    return registry.get(("classifier", model_name, weights_path),
                        lambda: RetrievalClassifier(model_name, weights_path))
