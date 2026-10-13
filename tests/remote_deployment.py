"""Staging/production verification against real services; no model substitutes.

Run inside the public Server host network. Credentials are read from a file and
never included in reports. Each run uses new collections and preserves evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['staging', 'production'], required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--verify-persistence', help='Report from before service restart')
    args = parser.parse_args()
    models = json.loads(Path(args.models).read_text())
    token = Path(args.token_file).read_text().strip()
    report = {'stage': args.stage, 'started_at': datetime.now(timezone.utc).isoformat(),
              'real_models': True, 'checks': [], 'collections': {}, 'requests': {}}
    def save():
        report['passed'] = sum(c['passed'] for c in report['checks'])
        report['failed'] = sum(not c['passed'] for c in report['checks'])
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    def record(name, passed, detail=None, seconds=None):
        entry = {'name':name, 'passed':bool(passed)}
        if detail is not None: entry['detail'] = detail
        if seconds is not None: entry['seconds'] = round(seconds, 3)
        report['checks'].append(entry); save()
        print(('PASS ' if passed else 'FAIL ') + name, flush=True)
    def request(client, path, payload):
        begin = time.monotonic()
        response = client.post(path, json=payload)
        try: data = response.json()
        except ValueError: data = {'body':response.text[:1000]}
        return response.status_code, data, time.monotonic()-begin
    def payload(language, collection):
        return {
            'query': '苍鹭项目的档案编号是什么？' if language == 'zh' else 'What is the archive code of Project Heron?',
            'collection':collection, 'search_method':'original',
            'embedding_model':models[language+'_embedding'], 'embedding_dim':768,
            'hyde_provider':'siliconflow' if language=='zh' else 'local',
            'hyde_model':models['qwen'] if language=='zh' else models['en_hyde'],
            'hyde_max_out_len':80, 'rerank_method':'bge' if language=='zh' else 'monot5',
            'rerank_model':models[language+'_reranker'], 'top_k':3, 'search_k':5,
            'compression_method':'llmlingua' if language=='zh' else 'recomp_extractive',
            'compression_model':models[language+'_compressor'], 'compression_ratio':0.8,
            'answer_model':models['qwen'], 'answer_max_out_len':256,
        }
    try:
        with httpx.Client(base_url=args.url, timeout=600, trust_env=False,
                          headers={'Authorization':'Bearer '+token}) as client:
            unauth = httpx.get(args.url+'/health', timeout=20, trust_env=False)
            record('unauthenticated requests rejected', unauth.status_code==401, unauth.status_code)
            record('authenticated Server health', client.get('/health').status_code==200)
            worker=httpx.get('http://127.0.0.1:8001/health', timeout=20, trust_env=False)
            record('SSH tunnel reaches real Worker', worker.status_code==200 and worker.json().get('mode')!='mock', worker.json())
            if args.verify_persistence:
                previous=json.loads(Path(args.verify_persistence).read_text())
                for language,p in previous['requests'].items():
                    for method in ['original','bm25']:
                        status,data,elapsed=request(client,'/api/query',{**p,'search_method':method})
                        record(f'{language}/{method}: data survives restart',status==200 and bool(data.get('retrieved_documents')),data,elapsed)
            else:
                docs={
                    'zh': ['苍鹭项目的档案编号为 HN-7429。该项目在蓝松实验室保存观测档案。',
                           '白石项目的档案编号为 WS-3168。该项目研究海岸沉积物。',
                           '红枫项目每周三维护设备。其负责人为李明。'],
                    'en': ['Project Heron has archive code HN-7429. Its observation records are stored at Blue Pine Laboratory.',
                           'Project White Stone has archive code WS-3168 and studies coastal sediments.',
                           'Project Red Maple performs equipment maintenance every Wednesday. Its coordinator is Alice.'],
                }
                for language in ['zh','en']:
                    collection='v010_'+args.stage+'_'+language+'_'+uuid.uuid4().hex[:8]
                    p=payload(language,collection);report['collections'][language]=collection;report['requests'][language]=p
                    begin=time.monotonic()
                    response=client.post('/api/index',data={'collection':collection,'embedding_model':p['embedding_model'],'embedding_dim':768},
                                         files=[('files',(f'{language}_{i}.txt',doc.encode(),'text/plain')) for i,doc in enumerate(docs[language])])
                    data=response.json()
                    ok=response.status_code==200 and data.get('document_count',0)>=3
                    record(language+': real indexing',ok,data,time.monotonic()-begin)
                    if not ok:continue
                    for method in ['original','bm25','hybrid','hyde','hyde_with_hybrid']:
                        status,data,elapsed=request(client,'/api/query',{**p,'search_method':method})
                        record(language+'/'+method+': retrieval, rerank, compression',status==200 and bool(data.get('retrieved_documents')) and bool(data.get('reranked_documents')) and bool(data.get('compressed_context')),data,elapsed)
                        if method in ['hyde','hyde_with_hybrid']:
                            record(language+'/'+method+': real HyDE output',status==200 and bool(data.get('hyde_document')))
                    status,data,elapsed=request(client,'/api/agent_query',p)
                    record(language+': grounded Qwen answer',status==200 and 'HN-7429' in data.get('answer',''),data,elapsed)
                if len(report['requests'])==2:
                    mixed={**report['requests']['en'],'query':'苍鹭项目的档案编号是什么？',
                           'hyde_provider':'siliconflow','hyde_model':models['qwen'],
                           'rerank_method':'bge','rerank_model':models['zh_reranker'],
                           'compression_method':'llmlingua','compression_model':models['zh_compressor'],
                           'search_method':'hyde_with_hybrid'}
                    status,data,elapsed=request(client,'/api/agent_query',mixed)
                    record('mixed Chinese/English models',status==200 and bool(data.get('answer')),data,elapsed)
                    p=report['requests']['zh']
                    for name,overrides in [('embedding identity mismatch',{'embedding_model':models['en_embedding']}),
                                           ('embedding dimension mismatch',{'embedding_dim':512}),
                                           ('classifier requires tuned weights',{'classification_enabled':True,'classification_model':models['classifier']}),
                                           ('removed profile rejected',{'profile':'zh'}),
                                           ('invalid adapter rejected',{'rerank_method':'unknown'})]:
                        status,data,elapsed=request(client,'/api/query',{**p,**overrides})
                        record(name,status==422,data,elapsed)
                    missing=dict(p);del missing['rerank_model']
                    status,data,elapsed=request(client,'/api/query',missing)
                    record('missing model rejected',status==422,data,elapsed)
                    def concurrent_call(item):return request(client,'/api/query',item)
                    requests=[report['requests']['zh'],report['requests']['zh'],report['requests']['en']]
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        results=list(pool.map(concurrent_call,requests))
                    record('concurrent same/different model requests complete',all(s==200 and bool(d.get('compressed_context')) for s,d,t in results),
                           [{'status':s,'seconds':round(t,3)} for s,d,t in results])
    except Exception as exc:
        record('run completed',False,type(exc).__name__+': '+str(exc).replace(token,'[REDACTED]'))
    report['completed_at']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps({'stage':args.stage,'passed':report['passed'],'failed':report['failed']},ensure_ascii=False),flush=True)
    return int(report['failed']>0)

if __name__=='__main__':raise SystemExit(main())
