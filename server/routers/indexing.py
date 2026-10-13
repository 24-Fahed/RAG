"""索引路由 - 处理文档上传和知识库构建。"""

import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool
import httpx
from pydantic import BaseModel, Field
from server.api_errors import INDEX_ERRORS

from server.services.pipeline import run_indexing_pipeline

router = APIRouter()


class IndexResponse(BaseModel):
    status: str = Field(description="成功时为 ok。")
    collection: str = Field(description="写入的集合名称。")
    document_count: int = Field(description="索引完成后集合的文档分块总数；不是本次上传文件数。")
    message: str = Field("", description="本次索引处理说明。")


@router.post("/index", response_model=IndexResponse, operation_id="index_documents",
             summary="上传文档并建立知识库索引", responses=INDEX_ERRORS,
             description="接收 multipart/form-data 文件，执行分块、向量化、Milvus 写入和 BM25 索引构建。集合、模型和维度必须显式指定。追加索引必须与已有集合的向量模型及维度一致。")
async def index_documents(
    files: list[UploadFile] = File(..., description="待上传文档；多个文件重复使用 files 字段。"),
    collection: str = Form(..., pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", description="目标集合名；必须显式指定。"),
    chunk_size: int = Form(512, gt=0, description="文档分块长度上限，按当前分块器的中英文词项规则计数。"),
    chunk_overlap: int = Form(20, ge=0, description="相邻分块重叠长度，必须小于 chunk_size。"),
    embedding_model: str = Form(..., min_length=1, pattern=r"\S", description="Embedding 模型 ID 或 Worker 本地路径；无默认模型。"),
    embedding_dim: int = Form(..., gt=0, description="模型实际输出维度，用于校验向量空间一致性。"),
):
    """上传文档并构建知识库。"""
    if chunk_overlap >= chunk_size:
        raise HTTPException(422, "chunk_overlap must be smaller than chunk_size")
    tmpdir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for f in files:
            path = os.path.join(tmpdir, os.path.basename(f.filename or "upload.txt"))
            with open(path, "wb") as out:
                content = await f.read()
                out.write(content)
            saved_paths.append(path)

        result = await run_in_threadpool(run_indexing_pipeline,
            data_path=tmpdir,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        return IndexResponse(
            status="ok",
            collection=collection,
            document_count=result.get("document_count", 0),
            message=result.get("message", ""),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(502, "Indexing service failed") from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
