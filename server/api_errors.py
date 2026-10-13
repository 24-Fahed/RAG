"""OpenAPI declarations for existing HTTP error responses."""
from pydantic import BaseModel, Field


class ServiceError(BaseModel):
    detail: str = Field(description="服务错误说明。")


class RequestError(BaseModel):
    detail: str | list[dict] = Field(description="业务校验失败时为字符串；请求结构校验失败时为错误数组，包含 loc、msg、type 等字段。")


INDEX_ERRORS = {
    422: {"model": RequestError, "description": "请求参数非法，或向量模型、维度与集合不一致。"},
    502: {"model": ServiceError, "description": "存储或推理服务调用失败；索引接口的上游 HTTP 超时也使用此状态码。"},
}
QUERY_ERRORS = {
    422: {"model": RequestError, "description": "字段缺失、未知字段、枚举或取值非法、阶段模型配置缺失，或向量空间不一致。"},
    502: {"model": ServiceError, "description": "检索或推理服务调用失败。"},
    504: {"model": ServiceError, "description": "推理 Worker HTTP 请求超时，包含排队等待时间。"},
}
