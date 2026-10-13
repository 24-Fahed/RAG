"""Validate and render the versioned PlantUML sources; no custom drawing code."""
import argparse
import subprocess
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--java', default='java')
parser.add_argument('--jar', required=True, help='Local PlantUML jar, tested with 1.2025.4')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sources = sorted((root / 'docs/v0.1.0/architecture').glob('*.puml'))
base = [args.java, '-Djava.awt.headless=true', '-jar', str(Path(args.jar).resolve()), '-charset', 'UTF-8']
subprocess.run(base + ['-checkonly', '-failfast2'] + [str(p) for p in sources], check=True)
for kind in ('png', 'svg'):
    subprocess.run(base + ['-t'+kind, '-failfast2'] + [str(p) for p in sources], check=True)
print(f'Validated and rendered {len(sources)} PlantUML sources to PNG and SVG.')

# Publish the same sources and PNGs used by the current documentation.
public = root / 'docs/architecture'
for source in sources:
    for path in (source, source.with_suffix('.png')):
        shutil.copy2(path, public / path.name)
