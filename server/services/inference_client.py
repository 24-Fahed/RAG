"""HTTP boundary: model identities and request settings are always explicit."""
from threading import Lock
import httpx
from server.config import INFERENCE_URL
from rag_langgraph.model_options import CLASSIFIER, EN_EMBEDDING, QWEN

_client: httpx.Client | None = None
_client_lock = Lock()


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            # Worker traffic uses the configured direct/private address or SSH
            # tunnel, never a workstation's outbound HTTP/SOCKS proxy.
            _client = httpx.Client(base_url=INFERENCE_URL, timeout=300.0, trust_env=False)
        return _client


def _post(path, payload, field):
    response = _get_client().post('/inference/' + path, json=payload)
    if response.status_code == 422:
        # Preserve model/parameter validation errors across the HTTP boundary.
        try:
            detail = response.json().get('detail')
        except (ValueError, AttributeError):
            detail = None
        raise ValueError(detail if isinstance(detail, str) else 'Invalid inference model or parameters')
    response.raise_for_status()
    return response.json()[field]


def classify(query: str, model_name: str = CLASSIFIER, weights_path: str | None = None) -> int:
    return _post('classify', locals(), 'label')


def embed(texts: list[str], model_name: str = EN_EMBEDDING, dim: int = 768) -> list[list[float]]:
    return _post('embed', locals(), 'embeddings')


def rerank(query: str, documents: list[dict], model: str = 'monot5', top_k: int = 10,
           model_name: str | None = None) -> list[dict]:
    return _post('rerank', locals(), 'documents')


def compress(query: str, context: str, method: str = 'recomp_extractive', ratio: float = 0.6,
             model_name: str | None = None) -> str:
    return _post('compress', locals(), 'compressed')


def hyde(query: str, max_out_len: int = 100, model_path: str = '/root/models/llama3-8b-ragga',
         provider: str = 'local') -> str:
    return _post('hyde', locals(), 'hypothetical_document')


def answer(query: str, context: str, model_name: str = QWEN, max_out_len: int = 1024) -> str:
    return _post('answer', locals(), 'answer')
