"""Inspect a built wheel without importing ComfyUI or optional dependencies."""
import argparse
import json
from pathlib import Path
import zipfile
p=argparse.ArgumentParser();p.add_argument('wheel',type=Path);args=p.parse_args()
with zipfile.ZipFile(args.wheel) as archive:
    names=set(archive.namelist());prefix='ComfyUI_Swwan/'
    required=['node_manifest.json','fonts/FreeMono.ttf','fonts/FreeMonoBoldOblique.otf',
              'licenses/MIT-rgthree.txt','licenses/GPL-3.0.txt','THIRD_PARTY_NOTICES.md',
              'web/js/swwan_seed.js','skills/comfyui-swwan-node-development/SKILL.md',
              'skills/comfyui-swwan-node-development/scripts/find_candidates.py',
              'examples/cpu-image-tools.json','examples/qwen2511-remove-single-swwan.json',
              'nodes/scheduling.py','ops/workflow_helpers.py']
    for filename in required:assert prefix+filename in names,filename
    assert not any('TTNorms' in n or '.DS_Store' in n for n in names)
    manifest=json.loads(archive.read(prefix+'node_manifest.json'))
    for row in manifest:assert prefix+row['module'].replace('.','/')+'.py' in names,row['id']
    assert len(manifest)==113
print('PASS: wheel source/manifest/asset/license/frontend/skill inventory')
