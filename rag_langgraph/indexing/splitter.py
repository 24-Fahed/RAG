"""
句子分割器。

在句子边界处将文档分割为块。
配置：chunk_size=512 词元，chunk_overlap=20 词元。
"""

import logging
import re

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    separator: str = " ",
) -> list[str]:
    """
    在句子边界处将文本分割为块。

    Args:
        text: 要分割的输入文本。
        chunk_size: 块的最大词元数（以词数近似）。
        chunk_overlap: 块之间重叠的词元数。
        separator: 词分隔符（默认：空格）。

    Returns:
        文本块列表。
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    # Slice original character spans: preserve Chinese punctuation and whitespace,
    # cap long sentences, count each CJK character as one approximate token.
    spans = list(re.finditer(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+", text))
    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(spans), step):
        stop = min(start + chunk_size, len(spans))
        chunks.append(text[spans[start].start():spans[stop - 1].end()])
        if stop == len(spans):
            break
    return chunks


def split_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> list[dict]:
    """
    将文档列表分割为块。

    Args:
        documents: 包含 'content' 和 'metadata' 的文档字典列表。
        chunk_size: 块的最大词元数。
        chunk_overlap: 块之间的重叠词元数。

    Returns:
        包含 'content'、'metadata' 和 'chunk_index' 的块字典列表。
    """
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc.get("metadata", {})
        chunks = split_text(content, chunk_size, chunk_overlap)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    **metadata,
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                },
            })

    logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


def _split_sentences(text: str) -> list[str]:
    """将文本分割为句子。"""
    # 处理常见的句子结尾
    sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _get_overlap_words(chunk_words: list[str], overlap_size: int, separator: str) -> str:
    """获取最后 N 个词用于重叠。"""
    if overlap_size <= 0:
        return ""
    overlap = chunk_words[-overlap_size:] if len(chunk_words) > overlap_size else chunk_words
    return separator.join(overlap)
