"""
嵌入模块。

使用 HuggingFace 模型生成 768 维向量嵌入。
默认模型：BAAI/bge-base-en-v1.5
"""

import logging


logger = logging.getLogger(__name__)


class Embedder:
    """使用 HuggingFace 模型的文本嵌入器。"""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", dim: int = 768):
        self.model_name = model_name
        self.dim = dim
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        actual_dim = self.model.config.hidden_size
        if actual_dim != dim:
            raise ValueError(f"Embedding dimension {dim} does not match model dimension {actual_dim}")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded embedding model: {model_name} (dim={dim})")

    def embed(self, text: str) -> list[float]:
        """为单个文本生成嵌入向量。"""
        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 [CLS] 词元嵌入
            embedding = outputs.last_hidden_state[:, 0]

        return embedding.squeeze().tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """为一批文本生成嵌入向量。"""
        import torch
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0]

            embeddings.extend(batch_embeddings.tolist())

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings


def get_embedder(model_name: str = "BAAI/bge-base-en-v1.5", dim: int = 768) -> Embedder:
    from rag_langgraph.models.runtime import registry
    return registry.get(("embedding", model_name, dim), lambda: Embedder(model_name, dim))
