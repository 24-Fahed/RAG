"""上下文压缩推理端点。

封装 rag_langgraph.models.compressors。
"""

from inference.schemas import CompressRequest

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.compressors import get_compressor

router = APIRouter()


class CompressResponse(BaseModel):
    compressed: str


@router.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest):
    compressor = get_compressor(method=req.method, model_name=req.model_name)
    result = compressor.compress(req.query, req.context, ratio=req.ratio)
    return CompressResponse(compressed=result)
