"""重排序推理端点。

封装 rag_langgraph.models.rerankers 的调度逻辑。
"""

from inference.schemas import Document, RerankRequest

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.rerankers import get_reranker

router = APIRouter()


class RerankResponse(BaseModel):
    documents: list[Document]


@router.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        return RerankResponse(documents=[])
    reranker = get_reranker(model_name=req.model, checkpoint=req.model_name)
    doc_texts = [d.content for d in req.documents]
    results = reranker.rerank(req.query, doc_texts)
    from collections import defaultdict, deque
    metadata = defaultdict(deque)
    for document in req.documents:
        metadata[document.content].append(document.metadata)
    for result in results:
        result["metadata"] = metadata[result["content"]].popleft()
    results = results[:req.top_k]
    return RerankResponse(
        documents=[Document(**d) for d in results]
    )
