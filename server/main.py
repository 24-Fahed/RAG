"""RAG server FastAPI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--mode",
    default="local",
    choices=["local", "staging", "production"],
    help="Deployment mode: local / staging / production (default: local)",
)
_args, _ = _parser.parse_known_args()

from server.config import load_config

load_config(_args.mode)

from fastapi import Depends, FastAPI
from server.auth import require_access_token

from runtime_diagnostics import install_runtime_diagnostics
from server.config import HOST, PORT
from server.routers import indexing, query

app = FastAPI(
    title="RAG Server API", version="0.1.0",
    dependencies=[Depends(require_access_token)],
    description="文档索引、检索与单轮回答接口。模型由请求逐项指定，无语言预设、默认模型组合或自动补全。模型标识与算法适配器分别配置。密钥由 Worker 本地凭证文件管理，不属于公开请求参数。",
    swagger_ui_parameters={"defaultModelExpandDepth": 3, "defaultModelsExpandDepth": 1},
    openapi_tags=[
        {"name": "query", "description": "知识库检索与无记忆单轮回答。"},
        {"name": "indexing", "description": "文档上传与知识库索引。"},
        {"name": "health", "description": "进程健康状态。"},
    ],
)
install_runtime_diagnostics(app, service_name="server", mode=_args.mode)

app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(indexing.router, prefix="/api", tags=["indexing"])


@app.get("/health", tags=["health"], summary="获取 Server 进程状态",
         description="仅表示 Server 进程可响应，不代表存储、Worker 或模型已就绪。",
         response_model=dict[str, str], operation_id="health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    reload = False
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    print(f"RAG Server starting at {HOST}:{PORT}")
    print(f"Config file: config/{_args.mode}.yaml")
    print(f"PID={os.getpid()} PPID={os.getppid()} reload={reload}")
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=reload)
