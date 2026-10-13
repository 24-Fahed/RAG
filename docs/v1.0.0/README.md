# v0.1.0 Notes

## Known Issues

### 1. Model compatibility

`llama3-8b-ragga` is an instruction-tuned chat model. After downgrading to `transformers==4.48.2`, the `apply_chat_template(..., return_tensors="pt")` path may return a tensor instead of a `BatchEncoding` object. This can cause compatibility issues if the code assumes `inputs["input_ids"]` is always available.

Reason:
- Different `transformers` versions return different input container types for chat-template generation.
- This release is focused on restoring the main retrieval and inference chain first.

Decision for this version:
- Record the compatibility risk.
- Leave broader compatibility hardening to the next version.

### 2. Empty answer output

`llama3-8b-ragga` may return an empty answer when it is prompted as a plain completion model.

Reason:
- This model is a fine-tuned instruct/chat model, not a plain causal completion model.
- A prompt like `Context ... Question ... Answer:` can lead to `generated_tokens=0`, even when retrieval, reranking, and compression all succeed.

Decision for this version:
- Record the issue and keep diagnostic logs.
- Move full prompt-format refinement and output stabilization to the next version.

### 3. Reranking can suppress relevant retrieved evidence

The current staging runs show that the retrieval layer can surface useful candidates, but the reranking stage may still push obviously relevant evidence down or replace it with only loosely related passages.

Observed pattern:
- `retrieved_documents` can already contain domain-relevant papers.
- `reranked_documents` may over-weight broad statistical language, country names, or generic biomedical patterns instead of preserving the strongest entity match.
- `repacked_context` and `compressed_context` then inherit that ranking error, because both stages are downstream of reranking.

Concrete examples from staging:

1. Query: `1 in 5 million in UK have abnormal PrP positivity.`
- Retrieval already surfaced clearly relevant prion documents such as:
  - `Biochemical Properties of Highly Neuroinvasive Prion Strains`
  - `Rapid and Sensitive RT-QuIC Detection of Human Creutzfeldt-Jakob Disease`
  - `Estimating prion concentration in fluids and tissues by quantitative PMCA`
- But reranking promoted less relevant material about:
  - patient survey reliability in England
  - BMI in UK adults
  - unrelated cancer papers
- This suggests the reranker is being distracted by surface signals like `UK`, prevalence-style numbers, and positivity/rate language instead of prioritizing the `PrP / prion` entity anchor.

2. Query: `5'-nucleotidase metabolizes 6MP.`
- Retrieval surfaced strong entity matches immediately:
  - `High Km soluble 5'-nucleotidase from human placenta`
  - `Prognostic importance of 6-mercaptopurine dose intensity in acute lymphoblastic leukemia`
- Reranking kept these documents near the top.
- This contrast suggests the current reranker behaves more reliably when the query contains uncommon, highly specific entity anchors.

Impact:
- End-to-end pipeline health can still look good in smoke tests.
- But factual claim-style quality degrades when reranking distorts an otherwise acceptable candidate set.

Working hypothesis:
- The dominant quality issue in these examples is not indexing failure.
- It is also not obviously a HyDE failure, because relevant documents are often present in the retrieved set.
- The most likely weak point is reranking behavior on claim-style biomedical queries, especially those containing:
  - ratios or prevalence language
  - country names
  - short factual assertions
  - entity + statistic combinations

Next-step focus:
- Compare rerankers (`monot5` vs `bge`) on the same claim set.
- Inspect whether HyDE expansion is adding noisy epidemiology language for factual claims.
- Measure not only retrieved hit rate, but also reranked hit rate and compressed-context retention.

## Why keep these items for the next version

This version is intended to stabilize the deployment chain first:
- server to inference connectivity
- model preload readiness
- indexing
- retrieval
- reranking
- compression

Generator prompt refinement and broader cross-version compatibility cleanup will be handled in the next release after the current chain is stable and reproducible.

## 多模型硬件资源调度（下一阶段）

v0.1.0 已实现全局模型初始化 FIFO 队列和每个模型实例的推理 FIFO 队列。同一模型串行，不同已加载模型允许并行；当前限单 Inference Worker 进程，尚无跨进程锁。

下一阶段根据 GPU 显存、内存与正在执行任务的预计峰值开销，判断硬件资源是否足够，再允许不同模型请求并发。增加资源预留、并发准入、模型驻留上限与卸载机制；不以固定模型数量替代实际资源判断。32 GB 显存并不保证任意组合及输入长度都可并发。
