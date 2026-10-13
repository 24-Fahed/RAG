"""Stage 1 local topology: real HTTP processes + Milvus, mock inference only.

Requires an isolated Milvus at localhost:19530 and free ports 8000/8001.
Creates unique collections and a temporary BM25 working directory; never uses
or deletes rag_collection. Run from the repository root.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
from pymilvus import MilvusClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='/tmp/rag-local-deployment-report.json')
    parser.add_argument('--bm25-only', action='store_true', help='Partial topology smoke; does not validate Milvus or indexing')
    args = parser.parse_args()
    checks = []
    processes = []
    streams = []
    collections = []
    prefix = 'local_v010_' + uuid.uuid4().hex[:10]
    milvus = None
    def query_request(collection, method, language='zh'):
        return {
            'query': '熊猫竹子' if language == 'zh' else 'pandas bamboo',
            'collection': collection, 'search_method': method,
            'embedding_model': f'BAAI/bge-base-{language}-v1.5', 'embedding_dim': 768,
            'hyde_provider': 'siliconflow' if language == 'zh' else 'local',
            'hyde_model': 'Qwen/Qwen2.5-7B-Instruct' if language == 'zh' else '/root/models/llama3-8b-ragga',
            'rerank_method': 'bge' if language == 'zh' else 'monot5',
            'rerank_model': 'BAAI/bge-reranker-v2-m3' if language == 'zh' else 'castorini/monot5-base-msmarco-10k',
            'compression_method': 'llmlingua' if language == 'zh' else 'recomp_extractive',
            'compression_model': 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank' if language == 'zh' else 'fangyuan/nq_extractive_compressor',
            'answer_model': 'Qwen/Qwen2.5-7B-Instruct',
        }
    def check(name, condition, detail=''):
        checks.append({'name':name, 'passed':bool(condition), 'detail':str(detail)[:500]})
        if not condition:
            raise AssertionError(f'{name}: {detail}')
    def free_port(port):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port))
    def start(module, port, directory):
        stream = open(Path(directory) / f'{module}-{len(processes)}.log', 'w')
        streams.append(stream)
        env = dict(os.environ, PYTHONPATH=str(ROOT), NO_PROXY='localhost,127.0.0.1')
        process = subprocess.Popen(
            [sys.executable,'-m','uvicorn',module+':app','--host','127.0.0.1','--port',str(port)],
            cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        processes.append(process)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f'{module} exited; see {stream.name}')
            try:
                response=httpx.get(f'http://127.0.0.1:{port}/health',timeout=1,trust_env=False)
                if response.status_code==200:return process
            except httpx.HTTPError:pass
            time.sleep(.1)
        raise RuntimeError(f'{module} startup timed out')
    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
    directory = tempfile.mkdtemp(prefix='rag-local-topology-')
    try:
        free_port(8000);free_port(8001)
        if not args.bm25_only:
            milvus=MilvusClient(uri='http://127.0.0.1:19530',timeout=20)
        start('inference.main',8000,directory)
        server=start('server.main',8001,directory)
        with httpx.Client(base_url='http://127.0.0.1:8001',timeout=60,trust_env=False) as client:
            health=httpx.get('http://127.0.0.1:8000/health',trust_env=False).json()
            check('Worker runs mock mode and is ready',health['mode']=='mock' and health['ready'])
            check('No models preloaded',health['preloaded_components']==[])
            for language in ('zh','en'):
                collection=prefix+'_'+language;collections.append(collection)
                model=f'BAAI/bge-base-{language}-v1.5'
                texts=['熊猫喜欢吃竹子。Pandas eat bamboo.', '汽车使用汽油。Cars use gasoline.', '数据库保存记录。Databases store records.']
                if args.bm25_only:
                    import pickle
                    bm25_dir=Path(directory)/'bm25_data'
                    bm25_dir.mkdir(exist_ok=True)
                    with (bm25_dir/(collection+'.pkl')).open('wb') as output:
                        pickle.dump({'documents':texts},output)
                else:
                    response=client.post('/api/index',data={'collection':collection,'embedding_model':model,'embedding_dim':768},
                        files=[('files',(f'{i}.txt',text.encode(),'text/plain')) for i,text in enumerate(texts)])
                    check(f'{language}: upload and index',response.status_code==200,response.text)
                    check(f'{language}: real Milvus rows',int(milvus.get_collection_stats(collection)['row_count'])>=3)
                methods=('bm25',) if args.bm25_only else ('original','hyde','hybrid','hyde_with_hybrid','bm25')
                for method in methods:
                    payload=query_request(collection, method, language)
                    response=client.post('/api/query',json=payload)
                    check(f'{language}/{method}: query status',response.status_code==200,response.text)
                    data=response.json()
                    check(f'{language}/{method}: retrieval and compression',bool(data['retrieved_documents']) and bool(data['compressed_context']))
                payload.update(search_method='bm25' if args.bm25_only else 'hyde_with_hybrid')
                response=client.post('/api/agent_query',json=payload)
                check(f'{language}: agent answer is mock',response.status_code==200 and response.json().get('answer','').startswith('[MOCK]'),response.text)
            mixed=query_request(prefix+'_en', 'bm25' if args.bm25_only else 'hyde_with_hybrid')
            mixed.update(embedding_model='BAAI/bge-base-en-v1.5', rerank_method='monot5',
                         rerank_model='castorini/monot5-base-msmarco-10k')
            response=client.post('/api/agent_query',json=mixed)
            check('Mixed Chinese/English selection',response.status_code==200,response.text)
            mixed['embedding_model']='BAAI/bge-base-zh-v1.5'
            if not args.bm25_only:
                check('Same dimension but different embedding model rejected',client.post('/api/query',json=mixed).status_code==422)
            check('Unknown adapter rejected',client.post('/api/query',json={**mixed, 'rerank_method':'typo'}).status_code==422)
            check('Untrained classification rejected',client.post('/api/query',json={**mixed, 'classification_enabled':True}).status_code==422)
            stop(server)
            start('server.main',8001,directory)
            for method in (('bm25',) if args.bm25_only else ('original','bm25')):
                response=client.post('/api/query',json=query_request(prefix+'_zh', method))
                check(f'{method}: persistence after Server restart',response.status_code==200 and bool(response.json().get('retrieved_documents')),response.text)
    except Exception as exc:
        checks.append({'name':'run completed','passed':False,'detail':str(exc)})
    finally:
        for process in reversed(processes):stop(process)
        for stream in streams:stream.close()
        if milvus is not None:
            for collection in collections:
                try:
                    if milvus.has_collection(collection):milvus.drop_collection(collection)
                except Exception as exc:
                    checks.append({'name':'cleanup '+collection,'passed':False,'detail':str(exc)})
            milvus.close()
    report={'stage':'local','topology':('partial: real HTTP + persisted BM25 + mock inference' if args.bm25_only else 'real HTTP + real Milvus + mock inference'),
            'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
            'logs':directory,'checks':checks}
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
    return int(report['failed']>0)


if __name__=='__main__':raise SystemExit(main())
