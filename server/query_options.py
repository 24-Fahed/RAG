"""Public query contract. Model selections are explicit and request-scoped."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelIdentifier = Annotated[str, Field(min_length=1, pattern=r"\S")]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: ModelIdentifier = Field(description='用户问题。必须包含非空白字符。')
    collection: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", description='目标知识库集合名。必须显式指定；向量检索要求与建库时使用相同的模型和维度。')
    search_method: Literal["original", "hyde", "hybrid", "hyde_with_hybrid", "bm25"] = Field(description='检索策略：original 为原问题向量检索；hyde 为扩展问题向量检索；hybrid 为原问题向量与 BM25 融合；hyde_with_hybrid 为 HyDE 向量与原问题 BM25 融合；bm25 为纯稀疏检索。')
    embedding_model: ModelIdentifier | None = Field(None, description='Embedding 模型 ID 或 Worker 本地路径。除 bm25 外必填，无默认模型；bm25 不使用此字段。')
    embedding_dim: int | None = Field(None, gt=0, description='Embedding 实际输出维度。除 bm25 外必填；用于一致性校验，不裁剪或转换向量。')
    hyde_provider: Literal["local", "siliconflow"] | None = Field(None, description='HyDE 调用方式。local 在 Worker 执行；siliconflow 调用远程服务。hyde 和 hyde_with_hybrid 策略必填。')
    hyde_model: ModelIdentifier | None = Field(None, description='HyDE 模型 ID 或 Worker 本地路径。hyde 和 hyde_with_hybrid 策略必填；须与 hyde_provider 相容。')
    hyde_max_out_len: int = Field(100, gt=0, le=8192, description='本次 HyDE 最大生成 token 数；仅在 HyDE 阶段使用，仍受模型或服务商限制。')
    rerank_method: Literal["monot5", "bge", "tilde", "rankllama"] = Field(description='重排适配器名称，决定推理实现；须与 rerank_model 的模型架构相容。')
    rerank_model: ModelIdentifier = Field(description='重排模型 ID 或 Worker 本地路径。必填，无默认模型；不接受以适配器名称代替模型标识。')
    top_k: int = Field(10, gt=0, le=1000, description='最终保留文档数上限，不得大于 search_k。')
    repack_method: Literal["sides", "compact", "compact_reverse"] = Field("sides", description='上下文重排方法：compact 按重排结果顺序拼接；compact_reverse 逆序拼接；sides 交替取首尾文档拼接。')
    compression_method: Literal["recomp_extractive", "recomp_abstractive", "llmlingua", "xprovence"] = Field(description='压缩适配器名称，决定推理实现；须与 compression_model 相容。')
    compression_model: ModelIdentifier = Field(description='压缩模型 ID 或 Worker 本地路径。必填，无默认模型。')
    compression_ratio: float = Field(0.6, gt=0, le=1, description='目标保留比例。实际长度取决于压缩适配器，不保证精确比例。')
    hybrid_alpha: float = Field(0.3, ge=0, le=1, description='混合检索中 BM25 的融合权重；向量权重为 1-alpha。仅 hybrid 和 hyde_with_hybrid 使用。')
    search_k: int = Field(100, gt=0, le=10000, description='各检索分支的候选数量上限；混合检索融合后保留至多该数量，再执行重排。')
    classification_enabled: bool = Field(False, description='是否执行检索必要性分类。关闭时直接继续检索；开启时必须提供分类模型及微调权重。')
    classification_model: ModelIdentifier | None = Field(None, description='分类基础模型 ID 或 Worker 本地路径。启用分类时必填，无默认模型。')
    classification_weights_path: ModelIdentifier | None = Field(None, description='检索必要性分类的微调权重路径，位于 Worker 文件系统。启用分类时必填。')
    answer_model: ModelIdentifier | None = Field(None, description='最终回答的 SiliconFlow 模型 ID。agent_query 必填；query 接受此字段但不执行回答阶段。无默认模型，不继承 hyde_model。')
    answer_max_out_len: int = Field(1024, gt=0, le=8192, description='最终回答最大生成 token 数，仅 agent_query 使用；仍受服务商限制。')

    @model_validator(mode="after")
    def validate_stages(self):
        if self.search_method != "bm25" and (self.embedding_model is None or self.embedding_dim is None):
            raise ValueError("Dense retrieval requires embedding_model and embedding_dim")
        if self.search_method in {"hyde", "hyde_with_hybrid"} and (self.hyde_provider is None or self.hyde_model is None):
            raise ValueError("HyDE requires hyde_provider and hyde_model")
        if self.classification_enabled and (self.classification_model is None or self.classification_weights_path is None):
            raise ValueError("Classification requires classification_model and fine-tuned classification_weights_path")
        if self.top_k > self.search_k:
            raise ValueError("top_k must not exceed search_k")
        return self

    def pipeline_kwargs(self):
        result = self.model_dump(exclude={"answer_model", "answer_max_out_len"})
        result["collection_name"] = result.pop("collection")
        return result
