# Issue #7：中文完整原句压缩

生产代码新增可选 `xprovence` 适配器，使用 `naver/xprovence-reranker-bgem3-v1`，
固定 revision `8dd5d79b6b57b3498a1180352daaf8c457af832b`。
NAVER v1 是本轮四例小样本中的相对最佳候选，不代表全保真或回答全部正确。
代码接入不等于已经部署；LLMLingua2 入口保留，可显式切回。
该固定版本[官方模型卡](https://huggingface.co/naver/xprovence-reranker-bgem3-v1/blob/8dd5d79b6b57b3498a1180352daaf8c457af832b/README.md)
标注许可证为 **CC BY-NC-ND 4.0**，包含非商业和禁止演绎限制，不能当作宽松商用许可。
公共 API 和 CLI 仍要求显式指定方法及模型，通用默认比例不变；中文请求示例为：

```json
{
  "compression_method": "xprovence",
  "compression_model": "naver/xprovence-reranker-bgem3-v1",
  "compression_ratio": 0.8
}
```

生产与实验共享 `rag_langgraph/models/xprovence.py` 的分块与评分实现。
沿用 NAVER 问题规范化及特殊 token 格式，从 forward 的 `compression_logits`
取得 token keep 概率，按每个原句的 token 均值评分。采用实验相同的
`select_sentences`：概率降序、同分按原句顺序，达到预算后按来源顺序输出完整原句。
预算固定使用 `Qwen/Qwen2.5-7B-Instruct` tokenizer，revision
`a09a35458c702b33eeacc393d103063234e8bc28`；仅下载 tokenizer 资源，不下载 Qwen 权重。
`.8` 的目标是 `ceil(source_tokens * .8)`，不是模型概率阈值；末句可超预算，
短文也可能完全不缩短。整句保留并不保证跨句关系、否定、限定和来源事实全部保留。

中文标点和换行分句，HTTP(S) URL 内标点不分句，输出使用原文切片。
固定 checkpoint 的本地 snapshot 也用于内嵌 tokenizer，避免浮动版本。
长文按模型窗口分块完整评分；问题过长、超窗原句、异常概率等直接报错，不静默截断或返回原文。
模型加载需执行已固定 revision 的自定义代码。Zilliz 与 NAVER v2 仍保留为实验候选，
不注册生产适配器。

推理依赖最小更新为 `transformers==4.57.1`，其 tokenizers 等传递依赖随之解析；
保留已有 torch、sentence-transformers 和 LLMLingua 约束。
新增 `spacy==3.7.5`，且 NAVER 顶层代码导入时必须能加载 `xx_sent_ud_sm==3.7.0`，
即使本适配器不用其分句也不能省略。优先复用已验证的本地 wheel：

```bash
python -m pip install /path/to/xx_sent_ud_sm-3.7.0-py3-none-any.whl
```

需要补取时优先使用[国内镜像固定资源](https://hf-mirror.com/spacy/xx_sent_ud_sm/resolve/353872c5d94a8a4319df108a87499babc7e23bc7/xx_sent_ud_sm-any-py3-none-any.whl)，
必要时回退同一 revision 的[官方资源](https://huggingface.co/spacy/xx_sent_ud_sm/resolve/353872c5d94a8a4319df108a87499babc7e23bc7/xx_sent_ud_sm-any-py3-none-any.whl)。
生产模型及预算 tokenizer 默认使用 `https://hf-mirror.com`，可通过 `HF_ENDPOINT` 显式覆盖。
本地轻量契约测试不能证明真实依赖兼容。独立环境的
[003 真实验证记录](../../../deployments/issue7-api-validation-003.json)
已通过进程内 FastAPI TestClient 调用真实 Worker 路由、ModelRegistry/FIFO 和 GPU 模型：
NAVER v1 四例均返回 HTTP 200，输出 hash 全部等于修复后 002 的 v1；
在 Transformers 4.57.1 下，既有中文 BGE embedding 返回两组 768 维有限数值向量，
BGE rerank 返回有限分数且相关文档排首位。
旧 LLMLingua 已在相同 Transformers 4.57.1 的 001/002 实验中运行，
尚未单独完成生产回滚演练。003 未部署生产服务，也不证明回答质量全部合格。

## 验证入口

轻量契约测试：

```bash
python -m pytest tests/unit/test_xprovence.py tests/unit/test_semantic_highlight.py tests/unit/test_compression_candidates.py tests/unit/test_models.py tests/integration/test_api.py -q
```

真实模型对照现为五组：原文、旧 LLMLingua2、Zilliz 双语、NAVER XProvence v1、NAVER XProvence v2。
NAVER 两组的实验入口仍为 `tests/compression_candidates.py`，从正式模块导入共享评分实现；生产只注册 v1。
固定版本分别为 `8dd5d79b6b57b3498a1180352daaf8c457af832b`、
`f2798d357213c40164bdd79557d6d2e43ca7d8d2`。
三候选均按同一回答 tokenizer 的 `ceil(source_tokens * 0.8)` 预算选完整原句，
记录实际 token 数与超预算量；旧组仍使用原生 LLMLingua2 rate，不能当作等预算候选。
NAVER 原生 process 不返回句概率；实验从 forward 的 compression_logits 提取 keep 概率，
按原句 token 均值评分，替代原生 process 的双阈值 rounding，且输出始终来自原文。
沿用 NAVER 的问题规范化和特殊 token 格式；原句按统一中文规则切分，不用其 spaCy 分句。
长文分块完整覆盖各句，超长单句明确拒绝；三模型原生窗口不同，证据中记录该差异。
固定 Zilliz checkpoint 的顶层 `max_length=512`，本轮实际按该窗口分块；
backbone 的 `max_position_embeddings=8194` 是位置容量配置，不代表默认运行窗口为 8k。
本轮不放大该窗口。单句超过扣除问题长度和保守 16 token 余量后的容量时直接报错，
即使整篇文档可以由多个短句分块处理，也不意味着支持任意超长单句。
这套评分与预算规则在实测前固定，不按结果择优改参。

真实模型对照需要 GPU/模型依赖、候选与旧模型权重、统一回答模型 tokenizer。
独立实验环境使用 Transformers 4.57.1；生产接入使用相同版本，升级前需验证共享模型依赖。NAVER 模型文件顶层仍导入 spaCy 并加载 `xx_sent_ud_sm`，
即使实验不使用其分句也必须提供该依赖。模型依次加载、卸载，避免同时占满显存。
先核实两个旧资源的完整 commit SHA，再运行；默认固定 ratio=0.8，不自动扫参：

```bash
python -m tests.compare_chinese_compression --output issue7-five-compression.json --baseline-revision <旧模型40位SHA> --tokenizer-revision <Qwen40位SHA>
python -m tests.answer_chinese_compression_remote --compression-evidence issue7-five-compression.json --output issue7-five-remote-answers.json
```

当前授权路径为独立 `answer_chinese_compression_remote`：只读取四例五组已完成压缩证据，
核对criteria输入和hash、五组完整性、没有已回答数据，再调用既有SiliconFlowGenerator。
本轮限定 `api.siliconflow.cn` 的 `Qwen/Qwen2.5-7B-Instruct`，最多20次请求，
统一 `max_tokens=1024`、`temperature=0.2`，无自动重试。每次请求之前及之后保存记录；
失败或中断也占一次预算并停止，不自动续跑。`in_flight_or_unknown` 可能已送达服务端，
须人工核对，不得直接重发。结果写新文件，记录压缩证据hash，拒绝覆盖旧结果。
本轮不使用比较脚本原有 `--answers` 便捷选项，以免绕过四例/20次的独立校验。
`answer_chinese_compression_local` 为已暂停的历史替代路径，本轮不下载或运行本地Qwen回答。
只跑压缩不能宣称回答合格；异常仅保存类型，不保存可能敏感的正文。
四例中文输入与独立评估标准见 `tests/issue7-quality-criteria.json`：
包括实体与别名对应、否定、限定、数字和未知起源；其中三例为本地重建，
一例来自历史 325 真实上下文，不等于精确复现用户原失败请求。

证据保存完整输入输出、hash、revision、依赖、参数、耗时和统一 tokenizer 的 token 数。
五组使用同一云回答模型及模板，provider revision不可见，未指定seed，不能宣称确定性。
既有生成适配器只返回回答文本，不提供usage和finish_reason，不能伪造这些服务端字段。
完整原句选取仍可能造成实际token保留率不同，必须结合保真与成本评估。
结果交独立评估人逐命题检查关系、否定和限定，并检查回答额外事实；
关键字存在、mock 通过、HTTP 200 或输出非空均不能证明效果合格。
