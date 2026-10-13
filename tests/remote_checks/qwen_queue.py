import json,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rag_langgraph.models.generator import get_generator
from rag_langgraph.models.remote_generator import SiliconFlowGenerator
original=SiliconFlowGenerator._chat
intervals=[]
def measured(self,*a,**kw):
 t=time.monotonic()
 try:return original(self,*a,**kw)
 finally:intervals.append({'thread':threading.current_thread().name,'start':t,'end':time.monotonic()})
SiliconFlowGenerator._chat=measured
model=get_generator('Qwen/Qwen2.5-7B-Instruct',provider='siliconflow')
barrier=threading.Barrier(5)
def invoke(i):
 barrier.wait()
 try:
  out=model.answer('档案号是什么？','档案编号 QA-3921',max_out_len=40) if i%2 else model.generate_hyde('图书馆文档检索',max_out_len=20)
  return {'passed':bool(out),'answer':out}
 except Exception as e:return {'passed':False,'error':type(e).__name__}
with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(invoke,range(5)))
intervals.sort(key=lambda x:x['start'])
checks=[{'name':'real Qwen HyDE and answer share serialized instance','passed':len(intervals)==5 and all(a['end']<=b['start'] for a,b in zip(intervals,intervals[1:]))},{'name':'five queued Qwen requests complete','passed':all(r['passed'] for r in results)}]
r={'checks':checks,'intervals':intervals,'results':results};Path('/root/autodl-tmp/rag-v010-20261013/qwen-queue.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(checks,flush=True)
