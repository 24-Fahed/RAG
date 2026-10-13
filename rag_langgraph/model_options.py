"""Shared model defaults; adapter choices and checkpoint identities are separate."""
EN_EMBEDDING = "BAAI/bge-base-en-v1.5"
ZH_EMBEDDING = "BAAI/bge-base-zh-v1.5"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER = "google-bert/bert-base-multilingual-cased"
RERANK_MODELS = {
    "monot5": "castorini/monot5-base-msmarco-10k",
    "bge": "BAAI/bge-reranker-v2-m3",
    "tilde": "ielab/TILDEv2-TILDE200-exp",
    "rankllama": "castorini/rankllama-v1-7b-lora-passage",
}
COMPRESSION_MODELS = {
    "recomp_extractive": "fangyuan/nq_extractive_compressor",
    "recomp_abstractive": "fangyuan/nq_abstractive_compressor",
    "llmlingua": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
}
