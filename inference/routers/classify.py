"""分类推理端点。

封装 rag_langgraph.models.classifier.RetrievalClassifier。
"""

from inference.schemas import ClassifyRequest

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.classifier import get_classifier

router = APIRouter()


class ClassifyResponse(BaseModel):
    label: int  # 0 = 无需检索, 1 = 需要检索


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.weights_path:
        from fastapi import HTTPException
        raise HTTPException(422, "Classification requires fine-tuned weights_path")
    classifier = get_classifier(
        model_name=req.model_name,
        weights_path=req.weights_path,
    )
    label = classifier.predict(req.query)
    return ClassifyResponse(label=label)
