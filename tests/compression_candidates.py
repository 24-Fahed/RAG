"""Comparison candidates; production and experiments share NAVER scoring."""
from rag_langgraph.models.semantic_highlight import (
    MODEL_ID, MODEL_REVISION, SemanticHighlightCompressor,
)
from rag_langgraph.models.xprovence import (
    XPROVENCE_CHECKPOINTS, XProvenceScorer, build_blocks,
)

CANDIDATES = {
    'zilliz_bilingual': {'model': MODEL_ID, 'revision': MODEL_REVISION},
    **XPROVENCE_CHECKPOINTS,
}


def load_candidate(name):
    if name == 'zilliz_bilingual':
        return SemanticHighlightCompressor(MODEL_ID)
    return XProvenceScorer(name)
