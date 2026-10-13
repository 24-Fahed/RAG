"""End-to-end HTTP orchestration with model/Milvus boundary substitutes.

The application routes, pipeline, HTTP inference client, model registry, vectorstore,
BM25 persistence, splitting and response validation all execute real code.
"""
import json
import sys
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rag_langgraph.model_options import EN_EMBEDDING, ZH_EMBEDDING, QWEN


@pytest.fixture
def system(monkeypatch, tmp_path, isolated_registry, inference_app):
    from rag_langgraph.indexing import embedding
    from rag_langgraph.models import generator, remote_generator, rerankers, compressors, classifier
    from server.services import inference_client, bm25_search
    from server.routers import query, indexing
    calls = []
    class Embedder:
        def __init__(self, model_name, dim): self.name, self.dim = model_name, dim
        def embed_batch(self, texts):
            calls.append(('embed', self.name, texts))
            return [[1.] + [0.] * (self.dim - 1) for t in texts]
    class Generator:
        def __init__(self, model_name): self.name = model_name
        def generate_hyde(self, query, max_out_len=100):
            calls.append(('hyde', self.name, max_out_len))
            return query + '检索扩展'
        def answer(self, query, context, max_out_len=1024):
            calls.append(('answer', self.name, query, context))
            return '依据知识库：' + context if context else '知识库证据不足'
    class Reranker:
        def __init__(self, model_name): self.name = model_name
        def rerank(self, query, docs):
            calls.append(('rerank', self.name))
            return [{'content': d, 'score':1., 'metadata':{}} for d in docs]
    class Compressor:
        def __init__(self, model_name): self.name = model_name
        def compress(self, query, context, ratio=.6):
            calls.append(('compress', self.name))
            return context
    class Classifier:
        def __init__(self, model_name, weights_path): self.name, self.weights = model_name, weights_path
        def predict(self, query):
            calls.append(('classify', self.name, self.weights))
            return 0
    monkeypatch.setattr(embedding, 'Embedder', Embedder)
    monkeypatch.setattr(generator, 'Generator', Generator)
    monkeypatch.setattr(remote_generator, 'SiliconFlowGenerator', Generator)
    monkeypatch.setattr(classifier, 'RetrievalClassifier', Classifier)
    for name in rerankers.RERANKER_REGISTRY: monkeypatch.setitem(rerankers.RERANKER_REGISTRY, name, Reranker)
    for name in compressors.COMPRESSOR_REGISTRY: monkeypatch.setitem(compressors.COMPRESSOR_REGISTRY, name, Compressor)
    collections = {}
    class MilvusClient:
        def __init__(self, **kwargs): pass
        def has_collection(self, name): return name in collections
        def create_collection(self, collection_name, schema):
            collections[collection_name] = {'schema':schema, 'data':[]}
        def prepare_index_params(self): return SimpleNamespace(add_index=lambda **kwargs:None)
        def create_index(self, **kwargs): pass
        def load_collection(self, name): pass
        def describe_collection(self, name): return collections[name]['schema']
        def insert(self, collection_name, data): collections[collection_name]['data'].extend(data)
        def flush(self, name): pass
        def get_collection_stats(self, name): return {'row_count':len(collections[name]['data'])}
        def search(self, collection_name, **kwargs):
            return [[{'entity':d, 'distance':.9} for d in collections[collection_name]['data'][:kwargs['limit']]]]
    def field(name, **kwargs): return {'name':name, 'params':kwargs}
    monkeypatch.setitem(sys.modules, 'pymilvus', SimpleNamespace(
        MilvusClient=MilvusClient, FieldSchema=field, CollectionSchema=lambda **kw:kw,
        DataType=SimpleNamespace(INT64=1,FLOAT_VECTOR=2,VARCHAR=3)))
    monkeypatch.setattr(bm25_search, '_instances', {})
    original = bm25_search.BM25Index
    monkeypatch.setattr(bm25_search, 'BM25Index', lambda: original(str(tmp_path)))
    app=FastAPI(); app.include_router(query.router,prefix='/api');app.include_router(indexing.router,prefix='/api')
    with TestClient(inference_app) as worker, TestClient(app) as server:
        monkeypatch.setattr(inference_client, '_get_client', lambda:worker)
        yield server, calls, collections


@pytest.mark.parametrize('language,model', [('zh',ZH_EMBEDDING),('en',EN_EMBEDDING)])
def test_index_query_and_agent(system, language, model, query_payload):
    client,calls,collections=system
    upload=client.post('/api/index',data={'collection':language,'embedding_model':model,'embedding_dim':768},files={'files':('source.txt','熊猫吃竹子。RAG retrieves evidence.'.encode())})
    assert upload.status_code==200, upload.text
    assert upload.json()['document_count']>0
    assert json.loads(collections[language]['schema']['description'])['embedding_model']==model
    for endpoint in ['query','agent_query']:
        response=client.post('/api/'+endpoint,json=query_payload(query='熊猫吃什么？', collection=language, embedding_model=model,
            hyde_provider='local' if language=='en' else 'siliconflow',
            hyde_model='/root/models/llama3-8b-ragga' if language=='en' else QWEN,
            rerank_method='bge' if language=='zh' else 'monot5',
            rerank_model='BAAI/bge-reranker-v2-m3' if language=='zh' else 'castorini/monot5-base-msmarco-10k',
            compression_method='llmlingua' if language=='zh' else 'recomp_extractive',
            compression_model='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank' if language=='zh' else 'fangyuan/nq_extractive_compressor'))
        assert response.status_code==200,response.text
        data=response.json()
        assert data['retrieved_documents'] and data['reranked_documents']
        assert data['reranked_documents'][0]['metadata']['chunk_index']==0
        assert '熊猫' in data['compressed_context']
        assert ('answer' in data)==(endpoint=='agent_query')
    assert all(c[1]==model for c in calls if c[0]=='embed')
    assert [c for c in calls if c[0]=='answer'][0][1]==QWEN


def test_mixed_models_and_collection_mismatch(system, query_payload):
    client,calls,_=system
    assert client.post('/api/index',data={'collection':'mixed','embedding_model':EN_EMBEDDING,'embedding_dim':768},files={'files':('source.txt',b'RAG retrieves evidence.')}).status_code==200
    payload=query_payload(query='中文查询', collection='mixed', embedding_model=EN_EMBEDDING, compression_model='custom/compressor', hyde_max_out_len=77)
    assert client.post('/api/agent_query',json=payload).status_code==200
    assert ('hyde',QWEN,77) in calls
    assert ('compress','custom/compressor') in calls
    assert ('rerank','castorini/monot5-base-msmarco-10k') in calls
    payload['embedding_model']=ZH_EMBEDDING
    assert client.post('/api/query',json=payload).status_code==422
    assert client.post('/api/index',data={'collection':'mixed','embedding_model':ZH_EMBEDDING,'embedding_dim':768},files={'files':('source.txt','中文'.encode())}).status_code==422


def test_classification_skip_and_empty_evidence(system, query_payload):
    client,calls,_=system
    response=client.post('/api/agent_query',json=query_payload(query='问候', classification_enabled=True, classification_model='custom/bert', classification_weights_path='tuned.pt'))
    assert response.status_code==200,response.text
    assert response.json()['classification_label']==0
    assert response.json()['answer']=='知识库证据不足'
    assert [c[0] for c in calls]==['classify','answer']
    response=client.post('/api/agent_query',json=query_payload(query='空库', collection='empty', search_method='original'))
    assert response.status_code==200,response.text
    assert response.json()['reranked_documents']==[]
    assert response.json()['answer']=='知识库证据不足'
