import json,time,statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import httpx
p=Path('/verification');base=json.loads((p/'production.json').read_text());token=(p/'token').read_text().strip()
report={'started_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':1800,'concurrency':3,'interval_seconds':5,'samples':[]}
start=time.monotonic()
with httpx.Client(base_url='http://127.0.0.1:8000',headers={'Authorization':'Bearer '+token},timeout=180,trust_env=False) as client:
    def request(lang):
        t=time.monotonic()
        try:
            r=client.post('/api/query',json={**base['requests'][lang],'search_method':'hybrid'})
            d=r.json();return {'language':lang,'status':r.status_code,'passed':r.status_code==200 and bool(d.get('compressed_context')),'seconds':time.monotonic()-t}
        except Exception as exc:return {'language':lang,'passed':False,'error':type(exc).__name__,'seconds':time.monotonic()-t}
    with ThreadPoolExecutor(max_workers=3) as pool:
        while time.monotonic()-start<1800:
            report['samples'].append({'elapsed':round(time.monotonic()-start,3),'requests':list(pool.map(request,['zh','zh','en']))})
            (p/'production-soak.json').write_text(json.dumps(report,indent=2))
            time.sleep(min(5,max(0,1800-(time.monotonic()-start))))
results=[r for sample in report['samples'] for r in sample['requests']];durations=sorted(r['seconds'] for r in results)
report.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,total=len(results),passed=sum(r['passed'] for r in results),failed=sum(not r['passed'] for r in results),p50_seconds=statistics.median(durations),p95_seconds=durations[min(len(durations)-1,int(len(durations)*.95))])
(p/'production-soak.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='samples'},flush=True)
