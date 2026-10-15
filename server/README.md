# PU-2 RAG Server

`server/` 是知识库服务入口，负责建库、检索和上下文整理编排。

## 访问认证

业务接口与 `/health` 要求 Bearer 令牌。管理员可通过 `python -m server.auth issue-token --subject friend --days 30` 生成熟人令牌，并按 ID 查看、续期或撤销；需先配置 `RAG_AUTH_TOKEN_DB_FILE` 的绝对持久化路径。现有 RS256 JWT 兼容。完整说明见[熟人访问令牌](../docs/auth.md)；这些令牌可以建库和查询，并非只读权限。

## 当前主流程

当前查询主流程为：

```text
query
-> hyde
-> embed
-> milvus search
-> bm25 search
-> hybrid fuse
-> rerank
-> repack
-> compress
```

说明：

- `classify` 当前部署中被旁路，默认所有请求进入检索流程
- `generate` 不属于当前项目提供的主流程能力

## 提供的接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/query` | 执行知识库查询 |
| `POST` | `/api/index` | 上传并索引文档 |
| `GET` | `/health` | 健康检查 |

## 查询请求模型

```python
class QueryRequest:
    query: str
    collection: str | None = None
    search_method: str = "hyde_with_hybrid"
    rerank_model: str = "monot5"
    top_k: int = 10
    repack_method: str = "sides"
    compression_method: str = "recomp_extractive"
    compression_ratio: float = 0.6
    hybrid_alpha: float = 0.3
    search_k: int = 100
```

## 查询响应模型

```python
class QueryResponse:
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    repacked_context: str
    compressed_context: str
    hyde_document: str | None
    classification_label: int | None
```

当前服务**不返回 `answer`**。

如果需要最终回答，应在服务外部额外接入语言模型，并使用本服务输出的上下文结果。

## 建库流程

```text
load documents
-> split
-> embed
-> store in milvus
-> build bm25
```

## 存储层

### Milvus

- 默认集合名：`rag_collection`
- 默认向量维度：`768`

### BM25

- 使用 `rank_bm25.BM25Okapi`
- 索引目录：`./bm25_data/`

## 启动方式

### 启动 Milvus

```bash
cd ../docker
docker-compose up -d
```

### 启动服务

```bash
python -m server.main --mode local
python -m server.main --mode staging
python -m server.main --mode production
```
