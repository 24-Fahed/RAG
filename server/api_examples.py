"""OpenAPI request examples only; never used to fill request fields at runtime."""

QUERY_EXAMPLES = {
    "explicit_models": {
        "summary": "显式指定模型的 JSON 请求体",
        "description": "仅用于展示请求格式。服务不以此示例补全任何请求参数。",
        "value": {
            "query": "检索增强生成如何工作？",
            "collection": "knowledge_zh",
            "search_method": "hyde_with_hybrid",
            "embedding_model": "BAAI/bge-base-zh-v1.5",
            "embedding_dim": 768,
            "hyde_provider": "siliconflow",
            "hyde_model": "Qwen/Qwen2.5-7B-Instruct",
            "hyde_max_out_len": 200,
            "rerank_method": "bge",
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "top_k": 5,
            "repack_method": "sides",
            "compression_method": "xprovence",
            "compression_model": "naver/xprovence-reranker-bgem3-v1",
            "compression_ratio": 0.8,
            "hybrid_alpha": 0.3,
            "search_k": 100,
            "classification_enabled": False,
            "classification_model": None,
            "classification_weights_path": None,
        },
    },
}
AGENT_QUERY_EXAMPLES = {
    "explicit_models": {
        "summary": "检索并生成单轮回答的 JSON 请求体",
        "description": "最终回答模型必须单独指定；示例不作为服务默认配置。",
        "value": {
            **QUERY_EXAMPLES["explicit_models"]["value"],
            "answer_model": "Qwen/Qwen2.5-7B-Instruct",
            "answer_max_out_len": 1024,
        },
    },
}
