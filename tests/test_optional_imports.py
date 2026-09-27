"""Registration succeeds when non-core optional libraries cannot import."""
import argparse
import importlib.abc
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.comfyui_root))
from comfy.cli_args import args as comfy_args
comfy_args.cpu=True
import nodes
from server import PromptServer
import asyncio
PromptServer(asyncio.new_event_loop())
OPTIONAL={'cv2','scipy','skimage','matplotlib','spandrel','color_matcher','triton','sageattention','nvvfx'}
for key in list(sys.modules):
    if key.split('.')[0] in OPTIONAL:del sys.modules[key]
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in OPTIONAL:raise ImportError('Blocked optional dependency: '+fullname)
sys.meta_path.insert(0,Blocked())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from migrate_qwen2511_workflow import load_nodes
reg=load_nodes(args.comfyui_root)
assert len(reg.NODE_CLASS_MAPPINGS)==113
for cls in reg.NODE_CLASS_MAPPINGS.values():cls.INPUT_TYPES()
assert not (OPTIONAL&{key.split('.')[0] for key in sys.modules})
print('PASS: registration and all schemas without optional libraries')
