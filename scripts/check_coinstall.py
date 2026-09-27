"""Load real third-party registries in both orders; no model inference.

Run in a ComfyUI environment with third-party dependencies already available.
Third-party import hooks run as they would during ComfyUI startup.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True)
p.add_argument('--plugin',action='append',type=Path,required=True)
p.add_argument('--reverse',action='store_true');p.add_argument('--output',type=Path)
args=p.parse_args();sys.path.insert(0,str(args.comfyui_root))
from comfy.cli_args import args as comfy_args
comfy_args.cpu=True
import nodes
from server import PromptServer
ROOT=Path(__file__).resolve().parents[1]
async def main():
    PromptServer(asyncio.get_running_loop())
    await nodes.init_extra_nodes(init_custom_nodes=False,init_api_nodes=False)
    paths=[ROOT,*args.plugin]
    if args.reverse:paths.reverse()
    registries={}
    for path in paths:
        before=set(sys.modules)
        if not await nodes.load_custom_node(str(path)):
            raise RuntimeError('Cannot load plugin: '+str(path))
        modules=[m for key,m in sys.modules.items() if key not in before and getattr(m,'__file__',None)==str(path/'__init__.py')]
        if not modules:raise RuntimeError('No registry found for '+str(path))
        registries[str(path)]=modules[-1].NODE_CLASS_MAPPINGS
    ours=registries[str(ROOT)]
    overlaps={path:sorted(set(ours)&set(mapping)) for path,mapping in registries.items() if path!=str(ROOT)}
    assert not any(overlaps.values()),overlaps
    for mapping in registries.values():
        # Other third-party plugins may overlap each other. Only Swwan ownership is checked.
        for key,cls in ours.items():assert nodes.NODE_CLASS_MAPPINGS[key] is cls,key
    result={'order':[str(x) for x in paths],'counts':{x:len(y) for x,y in registries.items()},'swwan_overlap':overlaps,'passed':True}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
asyncio.run(main())
