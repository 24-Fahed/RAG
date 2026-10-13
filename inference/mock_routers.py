"""模拟推理路由。

返回模拟数据，不加载任何机器学习模型。
用于 deploy.yaml 中 mode=mock 时的本地开发和测试。
"""

import random
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

MOCK_EMBEDDING_DIM = 768


from inference.schemas import (
    ClassifyRequest, EmbedRequest, RerankRequest, CompressRequest,
    HyDERequest, AnswerRequest, Document,
)
from inference.routers.classify import ClassifyResponse
from inference.routers.embed import EmbedResponse
from inference.routers.rerank import RerankResponse
from inference.routers.compress import CompressResponse
from inference.routers.generate import HyDEResponse

# ---------- 接口端点 ----------

@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    return ClassifyResponse(label=1)  # 始终返回"需要检索"


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embeddings = []
    for _ in req.texts:
        vec = [random.gauss(0, 1) for _ in range(req.dim)]
        norm = sum(x * x for x in vec) ** 0.5
        embeddings.append([x / norm for x in vec])
    return EmbedResponse(embeddings=embeddings, dim=req.dim)


@router.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    docs = req.documents[:req.top_k]
    scored = []
    for i, d in enumerate(docs):
        scored.append(Document(
            content=d.content,
            score=1.0 / (i + 1),
            metadata=d.metadata,
        ))
    return RerankResponse(documents=scored)


@router.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest):
    words = req.context.split()
    keep = max(1, int(len(words) * req.ratio))
    return CompressResponse(compressed=" ".join(words[:keep]))


@router.post("/hyde", response_model=HyDEResponse)
def hyde(req: HyDERequest):
    return HyDEResponse(
        hypothetical_document=f"Retrieval-augmented retrieval expands {req.query} with a short hypothetical passage to improve document search quality."
    )


@router.post("/answer")
def answer(req: AnswerRequest):
    return {"answer": f"[MOCK] {req.query}: {req.context}"}
