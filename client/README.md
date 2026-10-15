# PU-1 Client

`client/` 是命令行客户端，用来模拟上层调用方向知识库发起请求。

## 访问令牌

向管理员申请令牌，将完整字符串保存至本地文件。CLI 的 `--token-file /path/to/access.token` 放在 `query` 或 `index` 子命令之前；管理员续期后不需要修改文件，撤销后无法继续使用。新令牌和已有 JWT 使用方式相同。完整流程见[熟人访问令牌](../docs/auth.md)。本页下方旧命令与模型默认值属于历史示例，当前必填业务参数请以 `python -m client.client --help` 和 Server OpenAPI 为准。

它的职责是：

- 触发建库
- 提交查询
- 查看检索与上下文整理结果

它**不负责生成最终答案**。

## 查询接口

```http
POST {RAG_SERVER_URL}/api/query
```

### 请求字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | 必填 | 用户查询 |
| `collection` | string? | `null` | 集合名 |
| `search_method` | string | `hyde_with_hybrid` | 检索策略 |
| `rerank_model` | string | `monot5` | 重排序模型 |
| `top_k` | int | `10` | 最终保留文档数 |
| `repack_method` | string | `sides` | 上下文重打包策略 |
| `compression_method` | string | `recomp_extractive` | 上下文压缩方法 |
| `compression_ratio` | float | `0.6` | 压缩比例 |
| `hybrid_alpha` | float | `0.3` | 稀疏检索权重 |
| `search_k` | int | `100` | 初始召回深度 |

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `retrieved_documents` | `Document[]` | 初始召回结果 |
| `reranked_documents` | `Document[]` | 重排序结果 |
| `repacked_context` | string | 重打包后的上下文 |
| `compressed_context` | string | 压缩后的上下文 |
| `hyde_document` | string? | HyDE 生成的假设文档 |
| `classification_label` | int? | 分类标签 |

说明：

- 当前客户端不读取 `answer`
- 如果需要最终答案，需要额外接入语言模型消费这些输出

## 索引接口

```http
POST {RAG_SERVER_URL}/api/index
```

请求类型：`multipart/form-data`

### 请求字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `files` | `file[]` | 待上传文档 |
| `collection` | string | 集合名 |
| `chunk_size` | int | 分块大小 |
| `chunk_overlap` | int | 分块重叠大小 |

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 处理状态 |
| `collection` | string | 集合名 |
| `document_count` | int | 已入库分块数 |
| `message` | string | 附加信息 |

## 文档结构

```python
class Document:
    content: str
    score: float
    metadata: dict
```

## 常用命令

### 单次查询

```bash
python -m client.client --mode staging query "What is RAG?"
```

### 交互式查询

```bash
python -m client.client --mode staging query --interactive
```

### 带参数查询

```bash
python -m client.client --mode staging query "What is RAG?" \
  --search-method hyde_with_hybrid \
  --rerank-model monot5 \
  --repack-method sides \
  --compression-method recomp_extractive \
  --compression-ratio 0.6
```

### 上传目录建库

```bash
python -m client.client --mode staging index ./documents --collection rag_collection
```
