"""Run in the deployed Server container; never print tokens or key material."""
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx
from joserfc import jwt
from joserfc.jwk import RSAKey
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from server import auth
from server.main import app

checks=[]
def check(name,ok):
    checks.append({'name':name,'passed':bool(ok)})
    print(('PASS ' if ok else 'FAIL ')+name,flush=True)
private=RSAKey.import_key(auth.load_key('RAG_AUTH_PRIVATE_KEY_FILE'))
valid=auth.generate_token('bearer-verification',300)
claims=dict(jwt.decode(valid,RSAKey.import_key(auth.load_key('RAG_AUTH_PUBLIC_KEY_FILE'))).claims)
def signed(**updates):return jwt.encode({'alg':'RS256','typ':'at+jwt'},{**claims,**updates},private)
parts=valid.split('.');parts[2]=('A' if parts[2][0]!='A' else 'B')+parts[2][1:]
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=15,trust_env=False) as c:
    for path in ['/health','/api/index','/api/query','/api/agent_query']:
        r=c.request('GET' if path=='/health' else 'POST',path)
        check(path+' requires Bearer',r.status_code==401 and r.headers.get('www-authenticate')=='Bearer')
    cases=[('valid','Bearer '+valid,200),('wrong scheme','Basic abc',401),('malformed','Bearer invalid',401),
           ('tampered signature','Bearer '+'.'.join(parts),401),
           ('expired','Bearer '+signed(exp=int(time.time())-120),401),
           ('wrong issuer','Bearer '+signed(iss='wrong'),401),
           ('wrong audience','Bearer '+signed(aud='wrong'),401),
           ('not yet valid','Bearer '+signed(nbf=int(time.time())+600),401),
           ('insufficient scope','Bearer '+signed(scope='other'),403)]
    for name,value,expected in cases:
        r=c.get('/health',headers={'Authorization':value});check(name,r.status_code==expected)
    schema=c.get('/openapi.json').json()
    check('OpenAPI declares bearer scheme',schema['components']['securitySchemes']['RAGAccessToken']['scheme']=='bearer')
    check('all API operations declare authentication',all(op.get('security')==[{'RAGAccessToken':[]}] for route in schema['paths'].values() for method,op in route.items() if method in ['get','post']))
# Isolated process-only failure injection; running server remains unchanged.
original=auth.load_key
def missing(_):raise auth.AuthConfigurationError('test')
auth.get_validator.cache_clear();auth.load_key=missing
try:
    auth.require_access_token(HTTPAuthorizationCredentials(scheme='Bearer',credentials=valid))
    check('missing key fails closed',False)
except HTTPException as e:check('missing key fails closed',e.status_code==503)
finally:auth.load_key=original;auth.get_validator.cache_clear()
with TemporaryDirectory() as folder:
    auth.init_keys(Path(folder));before=(Path(folder)/'private.pem').read_bytes()
    try:auth.init_keys(Path(folder));check('key initialization refuses overwrite',False)
    except ValueError:check('key initialization refuses overwrite',(Path(folder)/'private.pem').read_bytes()==before)
    check('private key permissions',((Path(folder)/'private.pem').stat().st_mode & 0o777)==0o600)
report={'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks)}
Path('/verification/bearer-auth.json').write_text(json.dumps(report,indent=2))
Path('/verification/auth-openapi.json').write_text(json.dumps(app.openapi(),ensure_ascii=False,indent=2)+'\n')
raise SystemExit(bool(report['failed']))
