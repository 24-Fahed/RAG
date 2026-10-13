import json,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
import rag_langgraph.indexing.embedding as module
from rag_langgraph.models.runtime import registry
out=Path('/root/autodl-tmp/rag-v010-20261013/queue-real.json')
events=[]; lock=threading.Lock(); loads=[]; calls=[]
Original=module.Embedder
class Measured(Original):
    def __init__(self,*a,**kw):
        start=time.monotonic()
        try:super().__init__(*a,**kw)
        finally:
            with lock:loads.append({'model':a[0],'start':start,'end':time.monotonic()})
    def embed_batch(self,*a,**kw):
        start=time.monotonic()
        try:return super().embed_batch(*a,**kw)
        finally:
            torch.cuda.synchronize()
            with lock:calls.append({'model':self.model_name,'thread':threading.current_thread().name,'start':start,'end':time.monotonic()})
module.Embedder=Measured
checks=[]
def check(name,ok):checks.append({'name':name,'passed':bool(ok)});print(name,ok,flush=True)
zh='BAAI/bge-base-zh-v1.5';en='BAAI/bge-base-en-v1.5'
with ThreadPoolExecutor(max_workers=4) as pool:
    models=list(pool.map(module.get_embedder,[zh,en,zh,en]))
check('cold different model loading intervals do not overlap',len(loads)==2 and sorted(loads,key=lambda x:x['start'])[0]['end']<=sorted(loads,key=lambda x:x['start'])[1]['start'])
check('same configuration cold requests reuse one instance',models[0] is models[2] and models[1] is models[3])
try:module.get_embedder(zh,512)
except ValueError:check('failed load propagates',True)
else:check('failed load propagates',False)
check('registry remains usable after failed load',module.get_embedder(zh) is models[0])
admissions=[]
condition=models[0]._queue._condition
original_wait=condition.wait_for
def observed_wait(predicate, timeout=None):
    # FIFOQueue calls wait_for while holding its condition, immediately after
    # assigning the ticket; observe the real implementation without replacing it.
    admissions.append({'thread':threading.current_thread().name,'ticket':models[0]._queue._next-1})
    return original_wait(predicate, timeout)
condition.wait_for=observed_wait
barrier=threading.Barrier(5)
def same(i):
    barrier.wait();return models[0].embed_batch(['真实并发队列验证。'*80]*8)
with ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(same,range(5)))
ordered=sorted(calls,key=lambda x:x['start'])
check('five real same-model calls never overlap',len(ordered)==5 and all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])))
check('same-model execution follows actual FIFO tickets',[c['thread'] for c in ordered]==[a['thread'] for a in sorted(admissions,key=lambda x:x['ticket'])])
condition.wait_for=original_wait
check('all same-model outputs have expected shape',all(len(r)==8 and len(r[0])==768 for r in results))
calls.clear();barrier=threading.Barrier(2)
def different(i):
    barrier.wait();return models[i].embed_batch(['Real parallel inference. '*80]*16)
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(different,range(2)))
check('different real-model calls overlap',len(calls)==2 and max(c['start'] for c in calls)<min(c['end'] for c in calls))
check('all different-model outputs have expected shape',all(len(r)==16 and len(r[0])==768 for r in results))
out.write_text(json.dumps({'real_models':True,'isolated_process':True,'checks':checks,'loading_intervals':loads,'same_model_intervals':ordered,'fifo_tickets':admissions,'different_model_intervals':calls,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks)},indent=2))
