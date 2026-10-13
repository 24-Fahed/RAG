# 已知问题

## 1. transformers torch.load 安全检查阻止模型加载

**状态**：已解决

**现象**：Inference Worker 加载 rerank 模型（`castorini/monot5-base-msmarco-10k`）时报错：

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`,
we now require users to upgrade torch to at least v2.6
```

**原因**：`transformers >= 4.51` 在 `modeling_utils.load_state_dict()` 中调用 `check_torch_load_is_safe()`，检查 torch 版本。AutoDL 的 torch 版本低于 2.6，触发该检查。

**调用链**：
```
rerank.py -> get_reranker() -> MonoT5Reranker.__init__()
  -> T5ForConditionalGeneration.from_pretrained()
    -> modeling_utils.load_state_dict()
      -> check_torch_load_is_safe()  ← 此处抛出异常
```

**已尝试的方案**（均失败）：
- 在 `inference/main.py` 中 monkey-patch `transformers.utils.import_utils.check_torch_load_is_safe`
  - 失败原因：`import transformers.utils.import_utils` 触发 transformers 全量初始化，但此时 `modeling_utils` 尚未被导入；待 rerank 路由首次被调用时，`modeling_utils` 才被导入并绑定原始函数引用
- 遍历 `sys.modules` 修补所有已加载模块
  - 失败原因：同上，`modeling_utils` 在修补时还未出现在 `sys.modules` 中

**可行的解决方案**（任选其一）：
1. **降级 transformers**：`pip install transformers==4.48.0`（不含此安全检查）
2. **升级 torch**：`pip install torch>=2.6`（满足安全检查要求）
3. **正确的 monkey-patch**：在 `rag_langgraph/models/rerankers.py` 中，在 `from transformers import ...` 之前插入 patch，或在 `transformers/modeling_utils.py` 源码中直接注释掉 `check_torch_load_is_safe()` 调用
4. **使用 safetensors 格式的模型**：错误信息提示 safetensors 格式不受此限制

**影响范围**：所有通过 `torch.load` 加载的 `.bin` 格式模型（rerank、compress 等），不影响 safetensors 格式的模型（embed、classify 已正常工作）

---

## 2. 分类器使用未微调的 BERT 基础模型

**状态**：已临时绕过

**现象**：分类器对所有查询返回 label 0（不需要检索），导致检索结果为空。

**原因**：`rag_langgraph/models/classifier.py` 加载 `google-bert/bert-base-multilingual-cased`，该模型没有经过检索必要性判断的微调。分类头权重为随机初始化（日志中 `classifier.weight | MISSING`）。

**当前处理**：在 `server/services/pipeline.py` 中注释掉分类步骤，强制 `label = 1`（所有查询走检索）。

**正确修复**：获取微调后的分类器权重文件，配置 `weights_path` 参数。

---

## 3. monkey-patch 时序问题

**状态**：已解决

**现象**：在 `inference/main.py` 中对 transformers 的 monkey-patch 对部分模块无效。

**原因**：Python 模块导入机制。`from transformers.utils.import_utils import check_torch_load_is_safe` 会在导入模块时绑定函数的本地引用。后续修改源模块的属性不会影响已绑定的引用。

正确的修补必须在目标模块（如 `modeling_utils`）导入之前完成，但 `import transformers.utils.import_utils` 会触发 `transformers.__init__` 的执行，导致其他子模块在 patch 之前就已经完成了导入和绑定。

---

## 4. Shell 脚本中文输出在 Linux 终端乱码

**状态**：已解决

**原因**：Windows 环境下编辑的 `.sh` 文件包含中文 echo 输出，通过 scp 传到 Linux 后编码不一致导致乱码。

**修复**：所有 `.sh` 脚本的 echo 输出改为英文，注释保留中文。

---

## 5. 三处模型单例忽略后续请求参数

**状态**：已解决

**现象**：模型实例创建后，后续请求即使指定不同的模型或参数，也继续使用首次创建的实例；该问题在串行请求中同样存在，并非仅由并发触发。

| 位置 | 被忽略的后续参数 | 影响 |
| --- | --- | --- |
| `rag_langgraph/indexing/embedding.py` 的 `get_embedder()` | `model_name`、`dim` | 请求切换中英文向量模型时仍使用原模型；此处 `dim` 是实例属性，不会改变模型实际输出维度 |
| `rag_langgraph/models/generator.py` 的 `get_generator()` | `model_path`、`max_out_len` | 继续使用原模型与首次设置的生成长度；默认启动预热设置 `max_out_len=100`，后续 HyDE 请求的长度参数不生效 |
| `rag_langgraph/models/classifier.py` 的 `get_classifier()` | `model_name`、`weights_path` | 继续使用原分类模型；后续指定的微调权重不会加载 |

**原因**：三个工厂函数仅在全局实例为 `None` 时使用传入参数创建对象，之后直接返回缓存实例，不检查参数是否匹配。生成长度保存在共享实例的 `self.max_out_len` 中，推理时没有独立接收本次请求的长度。

**影响范围**：`/inference/embed`、`/inference/hyde`、`/inference/classify`。即使请求返回成功，也可能未按本次指定的模型或参数执行。

**建议修复**：模型固定部署时，对不同模型或权重参数明确拒绝，避免静默忽略；需要多模型时，按模型路径及权重配置缓存实例并保护初始化。`max_out_len` 应作为每次推理的参数传入，不能通过修改共享实例属性实现请求间切换。

---

## 6. LLMLingua 接入与官方 API 不匹配

**状态**：已解决

**位置**：`rag_langgraph/models/compressors.py` 的 `LLMLinguaCompressor.__init__()` 和 `compress()`。

**问题**：

- 项目调用 `self.compressor.compress(...)`，核查时的官方 `PromptCompressor` 提供的是 `compress_prompt(...)`，未定义公开的 `compress()` 方法。使用该 API 的版本会在压缩调用处发生 `AttributeError`。
- 默认模型为 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`，但构造调用仅为 `PromptCompressor(model_name)`，没有传入 `use_llmlingua2=True`。官方实现的该参数默认值为 `False`，因此不会启用 LLMLingua-2 的初始化及压缩分支。
- `inference/requirements.txt` 未声明 `llmlingua`，项目也未固定其版本，不能保证部署环境具备兼容的依赖。

**影响范围**：使用 `compression_method="llmlingua"` 的请求。缺少依赖时会在导入阶段失败；安装不匹配版本时可能在接口调用或模型执行阶段失败。默认 `recomp_extractive` 压缩链路不受此项影响。

**建议修复**：选定并固定经过验证的 LLMLingua 版本，声明依赖；按该版本 API 使用 `compress_prompt()`，并为 LLMLingua-2 模型设置 `use_llmlingua2=True`。完成单请求压缩验证后再验证共享实例的并发行为。

**核查依据与限制**：[LLMLingua 官方源码](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py)。依据为官方 `main` 分支，不能据此断言所有历史版本行为相同；历史核查时本机未安装 LLMLingua；现有 GPU 实测见 [模型参数测试](v0.1.0/test-results/model-parameters.json)。
