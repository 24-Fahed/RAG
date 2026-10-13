import sys
from types import SimpleNamespace
from contextlib import nullcontext
import httpx
import pytest
from rag_langgraph.model_options import EN_EMBEDDING, ZH_EMBEDDING, QWEN


def test_model_factories_respect_all_identities(monkeypatch, isolated_registry):
    from rag_langgraph.indexing import embedding
    from rag_langgraph.models import classifier, generator, compressors, rerankers
    def fake(*args, **kwargs): return SimpleNamespace(args=args, kwargs=kwargs)
    monkeypatch.setattr(embedding, 'Embedder', fake)
    monkeypatch.setattr(classifier, 'RetrievalClassifier', fake)
    monkeypatch.setattr(generator, 'Generator', fake)
    monkeypatch.setitem(compressors.COMPRESSOR_REGISTRY, 'llmlingua', fake)
    monkeypatch.setitem(rerankers.RERANKER_REGISTRY, 'bge', fake)
    assert embedding.get_embedder(EN_EMBEDDING) is not embedding.get_embedder(ZH_EMBEDDING)
    assert embedding.get_embedder(EN_EMBEDDING, 768) is not embedding.get_embedder(EN_EMBEDDING, 512)
    assert classifier.get_classifier('bert', 'a') is not classifier.get_classifier('bert', 'b')
    assert generator.get_generator('a') is generator.get_generator('a')
    assert generator.get_generator('a') is not generator.get_generator('b')
    assert compressors.get_compressor('llmlingua', 'a') is not compressors.get_compressor('llmlingua', 'b')
    assert rerankers.get_reranker('bge', 'a') is not rerankers.get_reranker('bge', 'b')


def test_generator_request_length_not_shared(monkeypatch):
    from rag_langgraph.models.generator import Generator
    lengths = []
    class Tensor:
        shape = (1, 2)
        def __getitem__(self, item): return self
    model = SimpleNamespace(generate=lambda **kw: (lengths.append(kw['max_new_tokens']) or Tensor()))
    g = Generator()
    g.model = model
    g.tokenizer = SimpleNamespace(decode=lambda *a, **kw: 'passage')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(g, '_build_inputs', lambda q: ({'input_ids': Tensor()}, q, 2, False, False))
    assert g.generate_hyde('中文', 13) == 'passage'
    assert g.generate_hyde('English', 79) == 'passage'
    assert lengths == [13, 79]


def test_llmlingua2_contract(monkeypatch):
    from rag_langgraph.models.compressors import LLMLinguaCompressor
    seen = {}
    class PromptCompressor:
        def __init__(self, *, model_name, use_llmlingua2):
            seen.update(model=model_name, llmlingua2=use_llmlingua2)
        def compress_prompt(self, context, *, rate):
            seen.update(context=context, rate=rate)
            return {'compressed_prompt': '中文证据'}
    monkeypatch.setitem(sys.modules, 'llmlingua', SimpleNamespace(PromptCompressor=PromptCompressor))
    assert LLMLinguaCompressor('test').compress('问题', '中文没有空格', .4) == '中文证据'
    assert seen == {'model':'test', 'llmlingua2':True, 'context':'中文没有空格', 'rate':.4}


def test_remote_chat_protocol_and_no_memory(monkeypatch, tmp_path):
    from rag_langgraph.models import remote_generator
    requests = []
    real_client = httpx.Client
    def respond(request):
        import json
        requests.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={'choices':[{'message':{'content':' 回答 '}}]})
    credentials = tmp_path / 'secrets.yaml'
    credentials.write_text('siliconflow:\n  api_key: test-only-placeholder\n')
    monkeypatch.setattr(remote_generator, 'CREDENTIALS_FILE', credentials)
    monkeypatch.setattr(remote_generator.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(respond)))
    g = remote_generator.SiliconFlowGenerator(QWEN)
    assert g.generate_hyde('问题一', 20) == '回答'
    assert g.answer('问题二', '证据', 30) == '回答'
    headers, payload = requests[-1]
    assert headers['authorization'] == 'Bearer test-only-placeholder'
    assert payload['model'] == QWEN and payload['max_tokens'] == 30
    assert len(payload['messages']) == 2
    assert '问题一' not in str(payload)
    assert '证据' in payload['messages'][1]['content']


@pytest.mark.parametrize('body', [{}, {'choices':[]}, {'choices':[{'message':{'content':''}}]}])
def test_remote_malformed_response_is_error(monkeypatch, body, tmp_path):
    from rag_langgraph.models import remote_generator
    real_client = httpx.Client
    credentials = tmp_path / 'secrets.yaml'
    credentials.write_text('siliconflow:\n  api_key: test-only-placeholder\n')
    monkeypatch.setattr(remote_generator, 'CREDENTIALS_FILE', credentials)
    monkeypatch.setattr(remote_generator.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(200,json=body))))
    with pytest.raises(RuntimeError, match='completion failed'):
        remote_generator.SiliconFlowGenerator().answer('q','c')


def test_chinese_split_and_bm25(tmp_path):
    from rag_langgraph.indexing.splitter import split_text
    from server.services.bm25_search import BM25Index, tokenize
    text = '中文检索支持多种模型组合。' * 30
    chunks = split_text(text, chunk_size=20, chunk_overlap=4)
    assert len(chunks) > 10
    assert all(len(tokenize(chunk)) <= 20 for chunk in chunks)
    assert all(chunk in text for chunk in chunks)
    idx = BM25Index(str(tmp_path))
    idx.build(['熊猫吃竹子', '汽车使用燃油', '数据库存储记录'])
    assert idx.search('熊猫')[0]['content'] == '熊猫吃竹子'
    idx.save('test')
    loaded = BM25Index(str(tmp_path)); loaded.load('test')
    assert loaded.search('熊猫') == idx.search('熊猫')


@pytest.mark.parametrize('contents', [None, '', 'siliconflow: {}', 'siliconflow: {api_key: 123}', 'siliconflow: {api_key: [private-marker'])
def test_missing_or_invalid_credentials_fail_without_exposing_contents(monkeypatch, tmp_path, contents):
    from rag_langgraph.models import remote_generator
    credentials = tmp_path / 'secrets.yaml'
    if contents is not None:
        credentials.write_text(contents)
    monkeypatch.setattr(remote_generator, 'CREDENTIALS_FILE', credentials)
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'unused-environment-value')
    with pytest.raises(RuntimeError) as error:
        remote_generator.read_api_key()
    assert 'private-marker' not in str(error.value)
    assert 'unused-environment-value' not in str(error.value)


def test_credentials_file_overrides_environment_and_is_reloaded(monkeypatch, tmp_path):
    from rag_langgraph.models import remote_generator
    credentials = tmp_path / 'secrets.yaml'
    monkeypatch.setattr(remote_generator, 'CREDENTIALS_FILE', credentials)
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'unused-environment-value')
    for key in ('file-key-a', 'file-key-b'):
        credentials.write_text(f'siliconflow:\n  api_key: {key}\n')
        assert remote_generator.read_api_key() == key
