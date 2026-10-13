# RAG v0.1.0

RAG 提供中英文文档索引、检索、重排、压缩和无记忆单轮回答。查询请求分别指定各功能的模型及适配器，不提供语言 profile 或默认模型组合。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [架构视图](architecture/README.md) | 逻辑架构、程序单元、物理架构及三种部署模式 |
| [OpenAPI](api/openapi.json) | 从 FastAPI 生成的请求、响应及 Bearer 鉴权定义 |
| [已知问题](ISSUES.md) | 已解决问题及当前限制 |

## 系统组成

| 程序单元 | 职责 |
| --- | --- |
| PU-1 Client Application | 上传资料、提交查询、展示结果 |
| PU-2 RAG Server | 索引与查询编排，访问 Milvus 和 BM25 |
| PU-3 Inference Worker | 模型加载、推理调用、远程 Qwen 接入 |

同一实例的推理请求按 FIFO 排队，不同实例可以并行。所有模型初始化统一串行；模型按需加载后常驻，目前没有按显存容量限制跨模型并发或自动淘汰。

## HTTP 接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| POST | `/api/index` | 上传文档并建立索引 |
| POST | `/api/query` | 返回检索、重排和压缩结果 |
| POST | `/api/agent_query` | 检索后由 Qwen 组织单轮回答 |
| GET | `/health` | 服务进程健康状态 |

以上接口要求 `Authorization: Bearer <token>`。`/docs` 提供 Swagger UI，可通过 Authorize 输入 Token；`/openapi.json` 提供规范。Token 为 RS256 JWT，校验有效期、签发者、受众和权限。密钥存放于文件。

```bash
python -m server.auth init-keys --directory /secure/rag-auth
python -m server.auth generate-token --subject client --expires-in 3600
```

验证公钥与签发私钥分别通过 `RAG_AUTH_PUBLIC_KEY_FILE`、`RAG_AUTH_PRIVATE_KEY_FILE` 指定文件路径。CLI 使用 `--token-file /path/to/access.token`，Token 不进入模型参数。配置缺失不会自动关闭鉴权。

## 模型与存储

中文向量模型为 `BAAI/bge-base-zh-v1.5`，英文为 `BAAI/bge-base-en-v1.5`。同一集合的索引和查询必须使用相同向量模型及维度。其他阶段可以独立组合模型；参数定义见 OpenAPI。

Qwen 使用 `Qwen/Qwen2.5-7B-Instruct`，从 Worker 的 `inference/config/secrets.yaml` 读取 SiliconFlow API Key。检索必要性分类缺少微调权重，当前不启用。

## 部署模式

- local：本地程序与 Mock 推理。
- staging：公网 Server、GPU Worker 和真实模型联调。
- production：相同物理分工下的部署运行配置，不等同于项目已经业务上线。

历史未版本化文档归档于 `v0.0.0/`；各版本内部方案和测试记录保留在版本目录，不作为本页的使用文档。
