"""Identical request contracts for real and mock inference workers."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from rag_langgraph.model_options import CLASSIFIER, EN_EMBEDDING, QWEN


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(Request):
    query: str
    model_name: str = CLASSIFIER
    weights_path: str | None = None


class EmbedRequest(Request):
    texts: list[str]
    model_name: str = EN_EMBEDDING
    dim: int = Field(768, gt=0)


class Document(BaseModel):
    content: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class RerankRequest(Request):
    query: str
    documents: list[Document]
    model: Literal["monot5", "bge", "tilde", "rankllama"] = "monot5"
    model_name: str | None = None
    top_k: int = Field(10, gt=0)


class CompressRequest(Request):
    query: str
    context: str
    method: Literal["recomp_extractive", "recomp_abstractive", "llmlingua", "xprovence"] = "recomp_extractive"
    model_name: str | None = None
    ratio: float = Field(0.6, gt=0, le=1)


class HyDERequest(Request):
    query: str
    provider: Literal["local", "siliconflow"] = "local"
    model_path: str = "/root/models/llama3-8b-ragga"
    max_out_len: int = Field(100, gt=0, le=8192)


class AnswerRequest(Request):
    query: str
    context: str
    model_name: str = QWEN
    max_out_len: int = Field(1024, gt=0, le=8192)
