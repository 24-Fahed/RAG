# RAG v0.1.0

## 版本范围

v0.1.0 提供文档索引、中文及英文检索、按请求配置模型、模型加载与推理调度，以及基于检索证据的单轮回答。版本保留 Client、RAG Server、Inference Worker 三个可部署程序单元。

本目录为该版本的完整设计文档。`architecture/` 中的六份架构定义替代原 `docs/architecture/` 对应视图；原目录保留为历史基线。架构描述设计结构，实施与验收状态另行记录，不混入架构元素。

## 架构变更审核

[架构修改与重构清单](方案.md#architecture-changes) 集中列出相对提交 `71ce0fa` 的原结构、新结构、修改原因、影响范围、迁移要求及对应架构视图。

## 文档目录

| 文档 | 内容 |
| --- | --- |
| [技术方案](方案.md) | 原始需求追踪、设计决策、模块职责、接口及兼容性影响 |
| [架构定义](architecture/README.md) | 逻辑架构、程序单元、物理架构及三种部署模式；PlantUML 源码与渲染结果 |
| [接口说明](接口说明.md) | OpenAPI 规范、IDE 预览工具及在线文档入口 |
| [测试方案](测试方案.md) | local、staging、production 三个部署阶段的测试设计及准入条件 |
| [本地 HTTP 测试记录](test-results/local-http.json) | 独立进程 HTTP/BM25 局部链路的原始结果 |
| [部署测试记录](部署测试记录.md) | 服务器部署位置、真实模型联调结果、重启验证及剩余验收项 |

PR 文档暂不纳入本轮交付。

## 系统组成

| 程序单元 | 部署职责 | 主要实现 |
| --- | --- | --- |
| PU-1 Client Application | 上传文档，提交查询，展示结果 | `client/`；新增模型组合亦可直接通过 HTTP 调用 |
| PU-2 RAG Server | 参数解析、索引编排、检索编排、单轮回答编排；访问 Milvus 和 BM25 | `server/`、`rag_langgraph/indexing/`、`rag_langgraph/nodes/repacking.py` |
| PU-3 Inference Worker | 按参数选择模型，统一管理实例、加载队列和调用队列；调用 SiliconFlow | `inference/`、`rag_langgraph/models/` |

知识库使用 Milvus 保存向量，使用本地 BM25 索引执行稀疏检索。中文与英文 embedding 对应不同向量空间，索引和查询必须使用相同模型。Qwen 由 SiliconFlow 提供服务，不在 GPU 服务器加载其权重。

## 处理流程

- **索引：** 文档读取 → 中英文分块 → 指定模型向量化 → Milvus 写入及 BM25 构建。
- **query：** 可选分类 → 可选 HyDE → 检索与融合 → 重排序 → 重打包 → 压缩 → 返回检索结果。
- **agent_query：** 执行相同检索流程 → 将问题与压缩证据发送给 Qwen → 返回检索结果及 `answer`。不保存会话历史。

调用方逐项显式指定模型及对应适配器；接口不提供语言预设或默认模型组合。需求指定的模型清单见[技术方案 §2](方案.md#r2)。

## 部署与运行

| 模式 | 程序分布 | 模型运行方式 | 配置 |
| --- | --- | --- | --- |
| local | 开发机运行三个 PU 及存储服务 | Mock 推理，不调用真实 Qwen | 各程序 `config/local.yaml` |
| staging | 客户端、公网 Server、GPU Worker 分布部署 | 真实本地模型及 Qwen API，SSH 隧道通信 | 各程序 `config/staging.yaml` |
| production | 与 staging 相同的物理分工 | 最终配置、持久隧道、运行维护策略 | 各程序 `config/production.yaml` |

在仓库根目录、已安装对应依赖且 Milvus 可用的环境中启动本地服务：

```bash
python -m inference.main --mode local
python -m server.main --mode local
```

两个命令分别运行于独立终端。local 的 Worker 端口为 8000，Server 端口为 8001。通过 `/api/index` 建立集合后，向 `/api/query` 或 `/api/agent_query` 提交请求。API 定义见[OpenAPI 规范](api/openapi.json)，访问方式见[接口文档入口](接口说明.md)。

真实 Worker 限定为单进程，默认按需加载；`RAG_PRELOAD_MODELS=1` 开启预热。真实 Qwen 调用从 `inference/config/secrets.yaml` 的 `siliconflow.api_key` 读取密钥。模型下载默认使用国内镜像，可通过 `HF_ENDPOINT` 切换官方地址。基础 multilingual BERT 不含检索分类的微调权重，分类功能默认关闭。

## 访问鉴权

公开业务接口和健康检查使用 Bearer JWT 鉴权；密钥以文件保存。Token 签发、CLI `--token-file` 和 Swagger Authorize 使用方法见[接口说明](接口说明.md#bearer-token-鉴权)。

## 当前验证状态

| 验证项 | 状态 |
| --- | --- |
| staging / production 主检查 | 各 29 项通过 |
| 公网客户端问答 | 中英文均通过 |
| 数据保留及自动恢复 | 手动重启、进程异常退出恢复后的查询均通过 |
| 真实模型调度 | 加载串行、FIFO 调用、跨模型并行及 Qwen 共享队列验证通过 |
| 检索质量 | 中英文各 20 个事实问题的检索、重排及压缩均保留目标事实 |
| 持续运行 | 约 18 分 53 秒，621 次请求，1 次 502；按用户要求停止 |
| 已知运行问题 | Qwen 偶发超时；一次普通检索 502 原因未确定 |
| 分类模型 | 暂缓；本地测试及取消项目不再继续 |

服务已部署至公网服务器和 GPU 服务器，运行 production 配置。代码仍在本地 `feature/chinese-query-model-routing` 工作区，未提交、未合并；PR 暂不编写。完整证据与范围见[部署测试记录](部署测试记录.md)。
