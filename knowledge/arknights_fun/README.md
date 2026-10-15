# 明日方舟娱乐知识包

2026-10-15 的首版仓库快照：**61 条独立梗与黑话、3 条助手导航卡，共 64 张卡**。面向面试官或 HR 展示有来源、可检索、能说明边界的中文 RAG 查询。本轮只提供仓库资料和离线验证，没有写入线上知识库或部署服务。

助手自称“**RAG知识库查询Agent，了解明日方舟各种梗知识**”，并依据本包说明知识范围与可提供的帮助。它是本项目的查询助手，不代表鹰角官方，也不扮演游戏角色；不能从知识包获知用户私人身份。问“我是谁”时，应区分用户在问自己还是在请求助手自述。

## 内容与证据

| 类别 | 独立条数 | 例子 |
| --- | ---: | --- |
| 角色外号与台词 | 23 | kokodayo、莱茵拳皇、小火龙快乐位、灌铅抱枕 |
| 玩法黑话 | 21 | 调箱师、充电宝、低配与低练、绝食流、解手流 |
| 敌人与关卡 | 7 | 九号空降兵、掀地板、弑界杯、看流星雨 |
| 剧情二创 | 5 | 黑钢相声组、炭烤沙虫腿、游客B/C、黑心药厂 |
| 社区创作与数字梗 | 5 | 325、799、呼呼呼与哐哐哐、七星干员 |
| 助手导航 | 3 | 助手是谁、知识库有什么、能帮你做什么 |

同义词、数字变体和同一成句没有为了计数拆条。灌铅抱枕、购物袋打人等具有不同对象与独立释义，分别建卡。这里的“61条”包括娱乐相关黑话，不宣称每一条都是狭义网络迷因。

主要来源为 [PRTS 的封存黑话区](https://prts.wiki/w/明日方舟黑话·梗·成句)。站方明确提示其内容来自玩家、可能主观且不保证准确或及时；本包据此记录社区用法，不把社区调侃当成官方定论。没有采纳旧词条中的实时强度排名、历史漏洞攻略、未经独立核实的动机或首创者断言。

指定梗的处理：

- **325**：依据[创作者文章《仙术杯·梗百科》](https://www.bilibili.com/opus/1010752946069045251)第五届章节，只说明游戏结算数字，不把它直接当作赛事规则计算后的总分；未核对最初比赛录播与首发时间。
- **799**：依据[社区专条](https://akp.fandom.com/zh/wiki/799?variant=zh-tw)解释限定条件下的计算口径，不能推成提丰在所有场景的伤害。相关视频只核对页面信息，没有观看。
- **呼呼呼／哐哐哐**：仅收录提丰／莱伊粉丝的电器拟声与谐音层。证据包含[社区索引段落](https://moegirl.uk/index.php?title=呼呼呼&variant=zh-sg)及[创作者用例](https://www.bilibili.com/opus/1146216386396160039)，直接打开部分页面失败，读取类型在来源清单中逐项记录；未确认首次造梗帖。排除成人、性暗示以及露骨二创。

其他缺证、页面超时和明确排除项见 [pending.json](pending.json)。初版偏向早期社区文化，少量补入后续数字梗；尚未系统覆盖近年所有活动、终末地、跨游戏梗或梗的完整传播史。含剧情与关卡剧透。

## 文件与入库格式

| 文件 | 用途 |
| --- | --- |
| [corpus/documents.json](corpus/documents.json) | 唯一应导入的 UTF-8 JSON 对象数组 |
| [COVERAGE.md](COVERAGE.md) | 全部条目编号、类别与来源索引 |
| [sources.json](sources.json) | 60 个来源、读取方式、核实日期与证据边界 |
| [pending.json](pending.json) | 未收录、排除及仍需核实的事项；不导入 |
| [queries.json](queries.json) | 61 条标题冒烟与 27 条改写、消歧、导航查询 |
| [verify.py](verify.py) | 调用现有 loader、splitter 和真实 BM25 的离线验证 |
| [verification/bm25-report.json](verification/bm25-report.json) | 本次验证结果、语料 SHA-256、每问命中排名 |
| [request.example.json](request.example.json) | 将来联调时的显式中文查询请求，尚未在线运行 |

导入文件必须指定 `corpus/documents.json` 或只包含它的 `corpus/`。**不要导入整个 `knowledge/arknights_fun/`**，否则 loader 会把清单和请求示例也当作资料读取。当前 loader 不递归加载目录，也不识别 JSONL、网页或 PDF。

每条以 `content` 为正文；`id/title/category/aliases/source_ids` 等顶层字段由现有 loader 保留为元数据。不要额外包一层 `metadata`。别名、边界和主要来源同时写进正文，因为现有嵌入和 BM25 都只索引正文，重打包也只拼正文；BM25 的返回元数据为空，原文里的来源尤其重要。

默认 512/20 分块下每卡保持一块，前 200 字符各不相同，避免混合检索按前缀合并时误去重。导航卡使用“你是谁”“库里有什么”“你能帮我什么”等自然问法，但仍需通过真实模型联调检查重排与压缩后的表现。

## 离线运行与结果

在仓库根目录，使用安装了 `rank-bm25` 的 Python：

```powershell
python -m pip install -r tests/requirements.txt
python -X utf8 knowledge/arknights_fun/verify.py --report knowledge/arknights_fun/verification/bm25-report.json
```

可以复用已有测试虚拟环境，不需要重复安装。`verify.py` 不发 HTTP 请求，不读凭据，不下载模型，不连接 Milvus，不使用 Mock；它直接运行项目现有的加载、分块、分词及 `rank_bm25` 实现，仅在包内创建验证目录。详细 JSON 只在显式传入 `--report` 时写出。

本次结构检查全部通过：64 张卡、64 个分块、编号和标题唯一、来源引用均可解析、别名和主要来源位于正文、元数据保持扁平。61/61 标题冒烟、27/27 独立问法在 BM25 前 5 个候选中包含所有预期条目；标题冒烟不算独立质量证明，改写集也只是有限固定样例。

首轮导航召回曾发现问题，补充“我是谁”的消歧语句以及概述卡的推荐、新梗与起源覆盖说明后重跑通过。最终结果是**针对当前语料修订后的回归检查**，不是未见问题的盲测集。没有执行真实向量检索、模型重排、压缩、生成回答或在线验收，不能把离线候选召回写成端到端准确率。

## 以后如何导入与演示

下列是可执行的准备说明，本轮没有执行。先按[部署指南](../../DEPLOY_GUIDE.md)准备有中文模型的服务和有效访问凭据，确认客户端 `--mode` 对应的地址。这里的集合名只是建议的新快照名，实际使用前应确认尚未用于其他资料。

```powershell
python -m client.client --mode staging --token-file C:/secure/access.token index knowledge/arknights_fun/corpus/documents.json --collection arknights_fun_20261015_v1 --embedding-model BAAI/bge-base-zh-v1.5 --embedding-dim 768 --chunk-size 512 --chunk-overlap 20
```

当前索引是追加，重复上传不会按 `id` 更新或去重。先建立独立快照集合；修订后用新的集合名重新导入，避免把新旧版本混在一起。本包不包含删除集合、覆盖生产数据或迁移存储的操作。

确认入库后，客户端可以展示检索证据：

```powershell
python -m client.client --mode staging --token-file C:/secure/access.token query "799能说明提丰所有情况下的伤害吗" --collection arknights_fun_20261015_v1 --search-method hybrid --embedding-model BAAI/bge-base-zh-v1.5 --embedding-dim 768 --rerank-method bge --rerank-model BAAI/bge-reranker-v2-m3 --compression-method llmlingua --compression-model microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank --compression-ratio 1.0 --top-k 5 --search-k 30 --repack-method compact
```

最终回答在 Server `/docs` 里对 `POST /api/agent_query` 使用 [request.example.json](request.example.json)，通过 Authorize 设置已有凭据。请求中的中文嵌入模型与维度必须和入库一致；其他阶段模型也必须在 Worker 可用。示例先用原问题 `hybrid`，关闭检索必要性分类，不启用 HyDE。压缩比例 1.0 用于首轮观察信息保留，仍要检查模型实际输出，不能保证所有压缩器完全无损。

面试演示可以按以下顺序，用 3–5 分钟展示证据链：

1. “你是谁”“库里有什么”“你能帮我什么”：展示三张导航卡的召回、身份和真实范围。
2. “呼呼呼和哐哐哐各指谁”：展示拟声词、谐音别名到角色的映射。
3. “325是赛事总分吗”“799代表所有情况下的伤害吗”：展示数字检索、限定条件与不乱作推论。
4. “剑圣一定是闪灵吗”：展示同名消歧与多个证据条目。
5. “你知道我真实姓名吗”：正确行为是说明没有身份资料；这项要在生成阶段人工验收，离线候选召回不等于通过。

每次同时查看 `retrieved_documents`、`reranked_documents`、`compressed_context` 与 `answer`：先确认正确卡片是否进入候选，再确认重排、压缩是否保留限定条件及来源，最后检查回答有没有越过证据。若泛问稳定性仍不足，再基于实际失败讨论最小接入改动；本包没有改动核心检索框架。

## 扩充方式

新增条目必须有实质独立释义和已读取的来源，记录 `page_text/search_excerpt/video_page_metadata_only` 等实际读取类型。仅看见视频标题不能证明视频里的解释，更不能证明起源；来源不可读时留在待核实清单。社区观点、官方设定、实际机制与二创联想要分开。资料均以简短自写概括呈现，没有下载或打包图片、音频、视频或整页原文；来源作者与链接保留在卡片和清单中，源站的内容说明不由仓库许可证替代。

修订后更新范围卡的数量与日期、覆盖清单、来源及待核实清单，重跑离线验证，并另留未参与修订的新问题做端到端人工评估。优先补有直接创作用例的新角色、新活动与社区梗，不把同义词拆成多条来增加规模。
