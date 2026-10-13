"""基于 rank-bm25 的本地 BM25 搜索（纯 Python，无 Java 依赖）。

由 RAG 服务端用于稀疏检索。以 pickle 文件形式持久化，
与 Milvus 的 collection 并存。
"""

import os
import pickle
import re
from rank_bm25 import BM25Okapi

from server.config import BM25_INDEX_DIR


def tokenize(text: str) -> list[str]:
    # Character tokens for CJK plus words for Latin scripts: deterministic, no model download.
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9_]+", text.lower())


class BM25Index:
    """从文本块构建的 BM25 索引，持久化到磁盘。"""

    def __init__(self, index_dir: str = BM25_INDEX_DIR):
        self.index_dir = index_dir
        self.bm25: BM25Okapi | None = None
        self.documents: list[str] = []
        os.makedirs(index_dir, exist_ok=True)

    def build(self, documents: list[str]):
        tokenized = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        if self.bm25 is None:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {"content": self.documents[i], "score": float(scores[i]), "metadata": {}}
            for i in ranked[:top_k]
            if scores[i] > 0
        ]

    def save(self, collection_name: str):
        path = os.path.join(self.index_dir, f"{collection_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)

    def load(self, collection_name: str) -> bool:
        path = os.path.join(self.index_dir, f"{collection_name}.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        tokenized = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        return True

    @property
    def doc_count(self) -> int:
        return len(self.documents)


# 每个 collection 对应一个单例
_instances: dict[str, BM25Index] = {}


def get_bm25_index(collection_name: str) -> BM25Index:
    if collection_name not in _instances:
        idx = BM25Index()
        idx.load(collection_name)
        _instances[collection_name] = idx
    return _instances[collection_name]
