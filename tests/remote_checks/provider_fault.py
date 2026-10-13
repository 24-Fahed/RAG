import json
from pathlib import Path
import rag_langgraph.models.remote_generator as m
original=m.read_api_key
m.read_api_key=lambda:'invalid-verification-key'
checks=[]
try:
    m.SiliconFlowGenerator().generate_hyde('测试',max_out_len=8)
    checks.append({'name':'invalid key rejected','passed':False})
except RuntimeError as e:
    cause=e.__cause__;status=getattr(getattr(cause,'response',None),'status_code',None)
    checks.append({'name':'invalid key rejected without leaking credentials','passed':status in [401,403] and 'invalid-verification-key' not in str(e),'status':status,'error':str(e)})
m.read_api_key=original
try:
    text=m.SiliconFlowGenerator().generate_hyde('测试文档检索',max_out_len=20)
    checks.append({'name':'valid credentials work after failure','passed':bool(text),'output':text})
except Exception as e:checks.append({'name':'valid credentials work after failure','passed':False,'error':type(e).__name__})
Path('/root/autodl-tmp/rag-v010-20261013/provider-fault.json').write_text(json.dumps({'checks':checks},ensure_ascii=False,indent=2));print(json.dumps(checks,ensure_ascii=False))
