"""Retrieval and stateless answer orchestration use the same model selection."""
from typing import Annotated
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
import httpx

from server.query_options import QueryRequest
from server.api_errors import QUERY_ERRORS
from server.api_examples import QUERY_EXAMPLES, AGENT_QUERY_EXAMPLES
from server.services.pipeline import run_query_pipeline
from server.services import inference_client

router = APIRouter()


class DocumentResult(BaseModel):
    content: str = Field(description="文档分块正文。")
    score: float = Field(description="当前检索或重排阶段的分数；不同模型的分数不可直接比较。")
    metadata: dict = Field(default_factory=dict, description="文档元数据；具体字段由索引来源决定。")


class QueryResponse(BaseModel):
    retrieved_documents: list[DocumentResult] = Field(default_factory=list, description="重排前的检索结果，最多 top_k 条。")
    reranked_documents: list[DocumentResult] = Field(default_factory=list, description="重排后的文档，最多 top_k 条。")
    repacked_context: str = Field("", description="按 repack_method 重排并拼接的上下文；无证据时为空字符串。")
    compressed_context: str = Field("", description="压缩后的证据文本，供最终回答使用。")
    hyde_document: str | None = Field(None, description="HyDE 生成的假设文档；未执行时为 null，不作为回答证据。")
    classification_label: int | None = Field(None, description="0 表示跳过检索，1 表示继续检索；分类关闭时固定为 1，属于编排策略而非预测。")


class AgentQueryResponse(QueryResponse):
    answer: str = Field(description="根据当前问题和压缩证据生成的单轮回答。空证据仍进入回答阶段；不保存对话历史。")


def _run(req, agent=False):
    try:
        if agent and req.answer_model is None:
            raise ValueError("agent_query requires answer_model")
        result = run_query_pipeline(**req.pipeline_kwargs())
        if agent:
            result["answer"] = inference_client.answer(
                req.query, result["compressed_context"],
                model_name=req.answer_model, max_out_len=req.answer_max_out_len,
            )
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Inference worker timed out") from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(502, "Retrieval or inference service failed") from exc


@router.post("/query", response_model=QueryResponse, operation_id="query",
             summary="检索知识库并返回整理后的证据", responses=QUERY_ERRORS,
             description="执行可选分类、可选 HyDE、检索、重排、重打包及压缩。各阶段模型由请求显式指定，不提供语言预设或模型自动补全。未知字段返回 422。未启用阶段的参数不参与执行。")
def query(req: Annotated[QueryRequest, Body(
    description="JSON 请求体。Example Value 显示完整调用示例，Schema 显示字段及约束。",
    openapi_examples=QUERY_EXAMPLES,
)]):
    return _run(req)


@router.post("/agent_query", response_model=AgentQueryResponse, operation_id="agent_query",
             summary="检索知识库并生成单轮回答", responses=QUERY_ERRORS,
             description="接受与 query 相同的字段；额外要求 answer_model 非空，否则在执行检索前返回 422。检索结束后，将当前问题和 compressed_context 交给指定 SiliconFlow 模型组织回答。无会话记忆。")
def agent_query(req: Annotated[QueryRequest, Body(
    description="JSON 请求体。与 query 使用相同字段；本接口必须提供 answer_model。",
    openapi_examples=AGENT_QUERY_EXAMPLES,
)]):
    return _run(req, agent=True)
