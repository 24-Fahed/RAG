# 服务器验收检查

这些脚本仅用于本次已部署服务器；不启动本地 RAG 或本地 Milvus。

## 公网服务器

发布目录：`/home/ubuntu/rag-deploy/releases/v010-20261013`。

`verification/` 挂载到正式 Server 容器的 `/verification`。目录内包含测试专用短期令牌 `token`、模型参数 `model-paths.json`、主检查报告 `production.json`。令牌不写入报告或版本库。

| 脚本 | 内容 | 输出 |
| --- | --- | --- |
| `quality.py` | 中英文各 20 个事实、5 个无答案、3 个冲突案例，及两种混合组合的五种策略 | `production-quality.json` |
| `soak.py` | 30 分钟，每轮三个并发 hybrid 查询，轮间等待五秒 | `production-soak.json` |

示例：`sudo docker exec rag-server python /verification/quality.py`。

## GPU 服务器

发布目录：`/root/autodl-tmp/rag-v010-20261013`。脚本使用该目录的 `venv/bin/python`，`PYTHONPATH` 指向 `app/`。本地模型检查使用已有离线缓存 `hub/`。

| 脚本 | 内容 | 输出 |
| --- | --- | --- |
| `model_queues.py` | 独立进程内调用实际模型工厂，观察真实加载/调用区间及 FIFO 票号 | `queue-real.json` |
| `qwen_queue.py` | 五个真实 Qwen HyDE/answer 请求共享模型实例，检查调用区间 | `qwen-queue.json` |
| `provider_fault.py` | 隔离进程替换凭据读取返回值，验证无效密钥及有效密钥恢复；不修改凭据文件 | `provider-fault.json` |

区间测量只包装实际模型方法并记录时间，不替换队列算法或推理模型。不同模型重叠指调用时间区间重叠，不代表 CUDA 核函数必然同时执行。

所有失败保留在原始结果中。脚本执行完成与验收通过分别统计；无答案和冲突案例的自动断言需结合报告中的原始回答审查。
