import json,math
from pathlib import Path
import httpx
checks=[]
with httpx.Client(base_url='http://127.0.0.1:8001',timeout=180,trust_env=False) as c:
 vectors=[]
 for model in ['BAAI/bge-base-zh-v1.5','BAAI/bge-base-en-v1.5','BAAI/bge-base-zh-v1.5']:
  r=c.post('/inference/embed',json={'texts':['模型参数选择验证：图书馆中的文献检索。'],'model_name':model,'dim':768});r.raise_for_status();vectors.append(r.json()['embeddings'][0])
 checks.append({'name':'embedding identity actually changes vector','passed':sum(abs(a-b) for a,b in zip(vectors[0],vectors[1]))>1,'difference_l1':sum(abs(a-b) for a,b in zip(vectors[0],vectors[1]))})
 checks.append({'name':'switching back restores the same embedding output','passed':max(abs(a-b) for a,b in zip(vectors[0],vectors[2]))<1e-5})
 context='\n'.join(f'第{i}号项目的档案编号为 X{i:03d}，该项目负责图书馆文献整理和长期保存，同时记录借阅时间和设备维护计划。' for i in range(30))
 outputs=[]
 for ratio in [.2,.8,.2]:
  r=c.post('/inference/compress',json={'query':'档案编号有哪些？','context':context,'method':'llmlingua','model_name':'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank','ratio':ratio});r.raise_for_status();outputs.append(r.json()['compressed'])
 checks.append({'name':'cached LLMLingua respects per-request compression ratio','passed':len(outputs[0])<len(outputs[1]),'lengths':[len(x) for x in outputs]})
 checks.append({'name':'compression ratio does not leak between requests','passed':outputs[0]==outputs[2]})
r={'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks)};Path('/verification/model-parameters.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(r,flush=True)
