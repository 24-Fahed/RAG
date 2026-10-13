"""Export API documentation from the FastAPI app without starting services."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.main import app

output = ROOT / 'docs/v0.1.0/api'
output.mkdir(parents=True, exist_ok=True)
schema = app.openapi()
(output / 'openapi.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Generated {output / "openapi.json"}')
