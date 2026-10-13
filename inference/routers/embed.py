"""向量嵌入推理端点。

封装 rag_langgraph.indexing.embedding.get_embedder。
"""

from inference.schemas import EmbedRequest

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.indexing.embedding import get_embedder

router = APIRouter()


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embedder = get_embedder(model_name=req.model_name, dim=req.dim)
    embeddings = embedder.embed_batch(req.texts)
    return EmbedResponse(embeddings=embeddings, dim=len(embeddings[0]) if embeddings else 0)
