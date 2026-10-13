"""Create an isolated local-test Compose file without modifying existing stacks."""
import argparse
from pathlib import Path
import yaml

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='/tmp/rag-query-compose.yaml')
parser.add_argument('--minio-image',default='minio/minio:RELEASE.2023-03-20T20-16-18Z')
parser.add_argument('--embedded', action='store_true', help='Use Milvus embedded etcd and local object storage')
args=parser.parse_args()
if args.embedded:
    source={'services': {'milvus': {
        'image': 'milvusdb/milvus:v2.5.27',
        'command': ['milvus','run','standalone'],
        'environment': {'ETCD_USE_EMBED':'true', 'ETCD_DATA_DIR':'/var/lib/milvus/etcd',
                        'ETCD_CONFIG_PATH':'/milvus/configs/embedEtcd.yaml', 'COMMON_STORAGETYPE':'local'},
        'ports':['127.0.0.1:19530:19530','127.0.0.1:9091:9091'],
        'volumes':['milvus_data:/var/lib/milvus'],
        'healthcheck': {'test':['CMD','curl','-f','http://localhost:9091/healthz'],
                        'interval':'10s','timeout':'5s','retries':12,'start_period':'30s'},
    }},'volumes':{'milvus_data':None}}
    Path(args.output).write_text(yaml.safe_dump(source))
    print(args.output)
    raise SystemExit(0)
source=yaml.safe_load((Path(__file__).resolve().parents[1]/'docker/docker-compose.yml').read_text())
source.pop('version',None)
for service in source['services'].values():
    service.pop('container_name',None)
    if 'ports' in service:
        service['ports']=['127.0.0.1:'+port for port in service['ports']]
source['services']['minio']['image']=args.minio_image
Path(args.output).write_text(yaml.safe_dump(source))
print(args.output)
