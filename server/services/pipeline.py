"""Pipeline orchestration for the distributed retrieval-first RAG deployment."""

import json
import logging

from rag_langgraph.nodes.repacking import repacking_node
from server.config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT
from rag_langgraph.model_options import EN_EMBEDDING, CLASSIFIER
from server.services import inference_client
from server.services.bm25_search import get_bm25_index

logger = logging.getLogger(__name__)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _hybrid_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.3,
    k: int = 100,
) -> list[dict]:
    dense_scores = _normalize_scores([r["score"] for r in dense_results])
    sparse_scores = _normalize_scores([r["score"] for r in sparse_results])

    score_map: dict[str, dict] = {}
    for i, result in enumerate(dense_results):
        key = result["content"][:200]
        score_map[key] = score_map.get(
            key,
            {
                "content": result["content"],
                "dense": 0,
                "sparse": 0,
                "metadata": result.get("metadata", {}),
            },
        )
        score_map[key]["dense"] = dense_scores[i]

    for i, result in enumerate(sparse_results):
        key = result["content"][:200]
        if key not in score_map:
            score_map[key] = {
                "content": result["content"],
                "dense": 0,
                "sparse": 0,
                "metadata": result.get("metadata", {}),
            }
        score_map[key]["sparse"] = sparse_scores[i]

    fused = []
    for item in score_map.values():
        combined = alpha * item["sparse"] + (1 - alpha) * item["dense"]
        fused.append(
            {
                "content": item["content"],
                "score": combined,
                "metadata": item["metadata"],
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]




def run_query_pipeline(
    query: str,
    search_method: str = "hyde_with_hybrid",
    rerank_method: str = "monot5",
    top_k: int = 10,
    repack_method: str = "sides",
    compression_method: str = "recomp_extractive",
    compression_ratio: float = 0.6,
    hybrid_alpha: float = 0.3,
    search_k: int = 100,
    collection_name: str | None = COLLECTION_NAME,
    embedding_model: str = EN_EMBEDDING,
    embedding_dim: int = 768,
    hyde_provider: str = "local",
    hyde_model: str = "/root/models/llama3-8b-ragga",
    hyde_max_out_len: int = 100,
    rerank_model: str | None = None,
    compression_model: str | None = None,
    classification_enabled: bool = False,
    classification_model: str = CLASSIFIER,
    classification_weights_path: str | None = None,
) -> dict:
    """Run the retrieval-first query pipeline and return retrieval context outputs."""
    collection_name = collection_name or COLLECTION_NAME

    classification_label = 1
    if classification_enabled:
        if not classification_weights_path:
            raise ValueError("Classification requires fine-tuned weights")
        classification_label = inference_client.classify(
            query, model_name=classification_model, weights_path=classification_weights_path)
        if classification_label == 0:
            return {"retrieved_documents": [], "reranked_documents": [],
                    "repacked_context": "", "compressed_context": "",
                    "hyde_document": None, "classification_label": 0}

    hyde_document = None
    effective_query = query
    if search_method in ("hyde", "hyde_with_hybrid"):
        hyde_document = inference_client.hyde(query, model_path=hyde_model, provider=hyde_provider, max_out_len=hyde_max_out_len)
        effective_query = hyde_document

    dense_results = []
    if search_method in ("original", "hyde", "hybrid", "hyde_with_hybrid"):
        query_embedding = inference_client.embed([effective_query], model_name=embedding_model, dim=embedding_dim)[0]
        from rag_langgraph.indexing.vectorstore import MilvusVectorStore

        store = MilvusVectorStore(collection_name=collection_name, dim=embedding_dim, embedding_model=embedding_model,
                                  host=MILVUS_HOST, port=MILVUS_PORT)
        dense_results = store.search(query_embedding, top_k=search_k)

    sparse_results = []
    if search_method in ("hybrid", "hyde_with_hybrid", "bm25"):
        bm25 = get_bm25_index(collection_name)
        sparse_results = bm25.search(query, top_k=search_k)

    if search_method == "bm25":
        final_results = sparse_results
    elif search_method in ("hybrid", "hyde_with_hybrid"):
        final_results = _hybrid_fuse(dense_results, sparse_results, alpha=hybrid_alpha, k=search_k)
    else:
        final_results = dense_results

    reranked_documents = inference_client.rerank(query, final_results, model=rerank_method, top_k=top_k, model_name=rerank_model) if final_results else []
    repacked_context = repacking_node(
        {
            "reranked_documents": reranked_documents,
            "repack_method": repack_method,
        }
    )["repacked_context"]
    compressed_context = ""
    if repacked_context:
        compressed_context = inference_client.compress(
            query,
            repacked_context,
            method=compression_method,
            ratio=compression_ratio,
            model_name=compression_model,
        )

    return {
        "retrieved_documents": final_results[:top_k],
        "reranked_documents": reranked_documents,
        "repacked_context": repacked_context,
        "compressed_context": compressed_context,
        "hyde_document": hyde_document,
        "classification_label": classification_label,
    }


def run_indexing_pipeline(
    data_path: str,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    embedding_model: str = EN_EMBEDDING,
    embedding_dim: int = 768,
) -> dict:
    """Run indexing: load -> split -> embed -> store."""
    from rag_langgraph.indexing.loader import load_from_directory, load_from_json
    from rag_langgraph.indexing.splitter import split_documents

    import os

    if os.path.isdir(data_path):
        documents = load_from_directory(data_path)
    else:
        documents = load_from_json(data_path)

    if not documents:
        return {"document_count": 0, "message": "No documents found"}

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [chunk["content"] for chunk in chunks]

    embeddings = inference_client.embed(texts, model_name=embedding_model, dim=embedding_dim)

    from rag_langgraph.indexing.vectorstore import MilvusVectorStore

    store = MilvusVectorStore(collection_name=collection_name, dim=embedding_dim, embedding_model=embedding_model,
                                  host=MILVUS_HOST, port=MILVUS_PORT)
    metadata_strs = [json.dumps(chunk.get("metadata", {}), ensure_ascii=False) for chunk in chunks]
    store.insert(embeddings, texts, metadata_strs)
    store.flush()

    bm25 = get_bm25_index(collection_name)
    bm25.build(bm25.documents + texts)
    bm25.save(collection_name)

    stats = store.get_stats()
    return {
        "document_count": stats.get("row_count", len(texts)),
        "message": f"Indexed {len(texts)} chunks into {collection_name}",
    }
