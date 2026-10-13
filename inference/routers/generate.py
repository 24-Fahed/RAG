"""HyDE inference endpoint."""

from inference.schemas import HyDERequest, AnswerRequest

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.generator import get_generator

router = APIRouter()


class HyDEResponse(BaseModel):
    hypothetical_document: str


@router.post("/hyde", response_model=HyDEResponse)
def hyde(req: HyDERequest):
    generator = get_generator(
        model_path=req.model_path,
        provider=req.provider,
    )
    hypothetical = generator.generate_hyde(req.query, max_out_len=req.max_out_len)
    return HyDEResponse(hypothetical_document=hypothetical)


@router.post("/answer")
def answer(req: AnswerRequest):
    generator = get_generator(model_path=req.model_name, provider="siliconflow")
    return {"answer": generator.answer(req.query, req.context, max_out_len=req.max_out_len)}
