import json,time,uuid,re
from pathlib import Path
from datetime import datetime,timezone
import httpx
p=Path('/verification');base=json.loads((p/'production.json').read_text());token=(p/'token').read_text().strip();models=json.loads((p/'model-paths.json').read_text())
report={'started_at':datetime.now(timezone.utc).isoformat(),'checks':[],'collections':{}}
def record(name,passed,detail):
    report['checks'].append({'name':name,'passed':bool(passed),'detail':detail});report['passed']=sum(c['passed'] for c in report['checks']);report['failed']=sum(not c['passed'] for c in report['checks']);(p/'production-quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(('PASS ' if passed else 'FAIL ')+name,flush=True)
with httpx.Client(base_url='http://127.0.0.1:8000',headers={'Authorization':'Bearer '+token},timeout=360,trust_env=False) as client:
    def ask(payload,path='/api/agent_query'):
        start=time.monotonic()
        try:
            r=client.post(path,json=payload);return {'status':r.status_code,'body':r.json(),'seconds':round(time.monotonic()-start,3)}
        except Exception as e:return {'status':0,'error':type(e).__name__,'body':{},'seconds':round(time.monotonic()-start,3)}
    for lang in ['zh','en']:
        collection='v010_quality_'+lang+'_'+uuid.uuid4().hex[:8];report['collections'][lang]=collection
        docs=[];cases=[]
        for i in range(20):
            name=f'QX{i:02d}';code=f'AR-{8173+i*37}'
            doc=f'{name} 项目的唯一档案编号为 {code}。此编号已由档案管理员确认。' if lang=='zh' else f'Project {name} has the unique archive code {code}. The archivist has verified this code.'
            q=f'{name} 项目的档案编号是什么？' if lang=='zh' else f'What is the archive code of Project {name}?'
            docs.append(doc);cases.append(('fact',q,[code]))
        for i in range(5):
            q=f'QX{i:02d} 项目的预算金额是多少？' if lang=='zh' else f'What is the budget amount for Project QX{i:02d}?'
            cases.append(('unanswerable',q,[]))
        for i in range(3):
            name=f'CX{i:02d}';a=f'CX-{4100+i}';b=f'CX-{7100+i}'
            doc=f'{name} 项目两份同日记录互相矛盾：第一份的档案编号是 {a}，第二份是 {b}。没有优先级或更正说明。' if lang=='zh' else f'Two records of Project {name} from the same date contradict each other: one gives archive code {a}, the other {b}. Neither record has priority and no correction is available.'
            q=f'{name} 项目的档案编号能确定吗？' if lang=='zh' else f'Can the archive code of Project {name} be determined?'
            docs.append(doc);cases.append(('conflict',q,[a,b]))
        r=client.post('/api/index',data={'collection':collection,'embedding_model':base['requests'][lang]['embedding_model'],'embedding_dim':768},files=[('files',(f'{i}.txt',doc.encode(),'text/plain')) for i,doc in enumerate(docs)])
        record(lang+'/index',r.status_code==200,{'status':r.status_code,'body':r.json()})
        for i,(kind,q,expected) in enumerate(cases):
            payload={**base['requests'][lang],'collection':collection,'query':q,'search_method':'hybrid','top_k':3,'search_k':10,'compression_ratio':0.8}
            result=ask(payload);data=result['body'];answer=data.get('answer','');ok=result['status']==200
            if kind=='fact':ok=ok and all(s in answer for s in expected)
            elif kind=='conflict':ok=ok and (all(s in answer for s in expected) or bool(re.search('矛盾|冲突|无法确定|不确定|conflict|contradict|cannot|uncertain',answer,re.I)))
            else:ok=ok and bool(re.search('没有|未提供|未提|未给|无法|不能|不足|未包含|不清楚|not |no |cannot|insufficient|unknown|unavailable',answer,re.I)) and not re.search(r'[$￥¥]\s*\d',answer)
            result['expected']=expected;record(f'{lang}/{kind}/{i}',ok,result)
        # All retrieval strategies with alternate-language reranker and original-language compressor.
        other='en' if lang=='zh' else 'zh'
        for method in ['original','bm25','hybrid','hyde','hyde_with_hybrid']:
            payload={**base['requests'][lang],'collection':collection,'query':cases[0][1],'search_method':method,'rerank_method':'monot5' if other=='en' else 'bge','rerank_model':models[other+'_reranker']}
            result=ask(payload);data=result['body'];record(f'{lang}/mixed/{method}',result['status']==200 and 'AR-8173' in data.get('answer',''),result)
report['completed_at']=datetime.now(timezone.utc).isoformat();(p/'production-quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
