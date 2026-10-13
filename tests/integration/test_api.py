from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.query_options import QueryRequest
from rag_langgraph.model_options import EN_EMBEDDING, ZH_EMBEDDING, QWEN


def test_explicit_model_selection(query_payload):
    request = QueryRequest(**query_payload(embedding_model=ZH_EMBEDDING,
        rerank_method='bge', rerank_model='custom/reranker', compression_model='custom/compressor'))
    forwarded = request.pipeline_kwargs()
    assert forwarded['embedding_model'] == ZH_EMBEDDING
    assert forwarded['rerank_method'] == 'bge'
    assert forwarded['rerank_model'] == 'custom/reranker'
    assert forwarded['compression_model'] == 'custom/compressor'
    assert 'answer_model' not in forwarded


@pytest.mark.parametrize('field', ['collection', 'search_method', 'embedding_model', 'embedding_dim',
    'hyde_provider', 'hyde_model', 'rerank_method', 'rerank_model', 'compression_method', 'compression_model'])
def test_missing_model_configuration_rejected(field, query_payload):
    from pydantic import ValidationError
    payload = query_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        QueryRequest(**payload)


def test_inactive_stages_require_no_models(query_payload):
    payload = query_payload(search_method='bm25')
    for field in ['embedding_model', 'embedding_dim', 'hyde_model', 'hyde_provider', 'answer_model']:
        del payload[field]
    request = QueryRequest(**payload)
    assert request.embedding_model is None and request.hyde_model is None
    assert request.classification_model is None and request.answer_model is None


@pytest.mark.parametrize('extra', [
    {'top_k':0}, {'compression_ratio':1.1}, {'search_method':'typo'},
    {'classification_enabled':True}, {'profile':'zh'}, {'rerank_model_name':'old'}, {'rerank_model':'  '}, {'hyde_provider':'unknown'},
    {'collection':'../escape'}, {'hybrid_alpha':-1}, {'embedding_dim':0},
    {'top_k':200,'search_k':100}, {'embedding_modle':'typo'},
])
def test_query_validation(extra, query_payload):
    from server.routers.query import router
    app = FastAPI(); app.include_router(router, prefix='/api')
    with TestClient(app) as client:
        for route in ['query', 'agent_query']:
            assert client.post('/api/'+route, json=query_payload(**extra)).status_code == 422


def test_worker_routes_forward_model_and_request_parameters(monkeypatch, inference_app):
    from inference.routers import embed, rerank, compress, generate, classify
    seen = {}
    def factory(name, model):
        def load(*args, **kwargs):
            seen[name] = kwargs
            return model
        return load
    monkeypatch.setattr(embed, 'get_embedder', factory('embed', SimpleNamespace(embed_batch=lambda t:[[1.,2.]])))
    monkeypatch.setattr(rerank, 'get_reranker', factory('rerank', SimpleNamespace(rerank=lambda q,d:[{'content':x,'score':1.,'metadata':{}} for x in reversed(d)])))
    monkeypatch.setattr(compress, 'get_compressor', factory('compress', SimpleNamespace(compress=lambda q,c,ratio:c)))
    monkeypatch.setattr(classify, 'get_classifier', factory('classify', SimpleNamespace(predict=lambda q:1)))
    monkeypatch.setattr(generate, 'get_generator', factory('generate', SimpleNamespace(generate_hyde=lambda q,max_out_len:str(max_out_len), answer=lambda q,c,max_out_len:c)))
    with TestClient(inference_app) as client:
        assert client.post('/inference/embed',json={'texts':['中'],'model_name':'zh','dim':2}).status_code == 200
        assert seen['embed'] == {'model_name':'zh','dim':2}
        r = client.post('/inference/rerank',json={'query':'中','documents':[{'content':'a','metadata':{'id':1}},{'content':'b','metadata':{'id':2}}], 'model':'bge','model_name':'custom'})
        assert r.json()['documents'][0]['metadata'] == {'id':2}
        assert seen['rerank'] == {'model_name':'bge','checkpoint':'custom'}
        assert client.post('/inference/compress',json={'query':'q','context':'c','method':'llmlingua','model_name':'custom'}).status_code == 200
        assert seen['compress'] == {'method':'llmlingua','model_name':'custom'}
        for length in (17,80):
            r=client.post('/inference/hyde',json={'query':'中','provider':'siliconflow','model_path':QWEN,'max_out_len':length})
            assert r.json()['hypothetical_document'] == str(length)
        assert seen['generate'] == {'model_path':QWEN,'provider':'siliconflow'}
        assert client.post('/inference/classify',json={'query':'q'}).status_code == 422
        assert client.post('/inference/classify',json={'query':'q','model_name':'bert','weights_path':'fine.pt'}).status_code == 200
        assert seen['classify'] == {'model_name':'bert','weights_path':'fine.pt'}
        assert client.post('/inference/rerank',json={'query':'q','documents':[]}).json() == {'documents':[]}


def test_agent_preserves_retrieval_and_calls_answer_only_once(monkeypatch, query_payload):
    from server.routers import query
    results = {'compressed_context':'证据','classification_label':1}
    seen=[]
    monkeypatch.setattr(query,'run_query_pipeline', lambda **kwargs: dict(results))
    monkeypatch.setattr(query.inference_client,'answer',lambda q,c,**kw: seen.append((q,c,kw)) or '答案')
    app=FastAPI();app.include_router(query.router,prefix='/api')
    with TestClient(app) as client:
        assert 'answer' not in client.post('/api/query',json=query_payload()).json()
        response=client.post('/api/agent_query',json=query_payload())
        assert response.json()['answer']=='答案'
        assert response.json()['compressed_context']=='证据'
    assert len(seen)==1 and seen[0][2]['model_name']==QWEN


def test_upstream_failure_not_success(monkeypatch, query_payload):
    import httpx
    from server.routers import query
    def fail(**kwargs): raise httpx.ReadTimeout('timeout')
    monkeypatch.setattr(query,'run_query_pipeline',fail)
    app=FastAPI();app.include_router(query.router,prefix='/api')
    with TestClient(app) as client:
        assert client.post('/api/agent_query',json=query_payload()).status_code==504


def test_internal_worker_client_ignores_outbound_proxy(monkeypatch):
    from server.services import inference_client
    monkeypatch.setenv('ALL_PROXY', 'socks5://127.0.0.1:9')
    monkeypatch.setattr(inference_client, '_client', None)
    client = inference_client._get_client()
    try:
        assert client.trust_env is False
    finally:
        client.close()


def test_agent_requires_answer_model_before_pipeline(monkeypatch, query_payload):
    from server.routers import query
    calls = []
    monkeypatch.setattr(query, 'run_query_pipeline', lambda **kw: calls.append(kw) or {})
    app = FastAPI(); app.include_router(query.router, prefix='/api')
    payload = query_payload(); del payload['answer_model']
    with TestClient(app) as client:
        assert client.post('/api/agent_query', json=payload).status_code == 422
        assert calls == []
        assert client.post('/api/query', json=payload).status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize('upstream_status, expected_status', [(422, 422), (502, 502)])
def test_worker_validation_status_preserved(monkeypatch, query_payload, upstream_status, expected_status):
    import httpx
    from server.routers import query
    from server.services import inference_client
    def handler(request):
        return httpx.Response(upstream_status, json={'detail':'Embedding dimension mismatch'})
    with httpx.Client(transport=httpx.MockTransport(handler), base_url='http://worker') as worker:
        monkeypatch.setattr(inference_client, '_get_client', lambda: worker)
        def pipeline(**kwargs):
            inference_client.embed(['text'], model_name='requested-model', dim=512)
        monkeypatch.setattr(query, 'run_query_pipeline', pipeline)
        app=FastAPI(); app.include_router(query.router,prefix='/api')
        with TestClient(app) as client:
            response=client.post('/api/query',json=query_payload())
            assert response.status_code == expected_status
            if upstream_status == 422:
                assert response.json()['detail'] == 'Embedding dimension mismatch'
