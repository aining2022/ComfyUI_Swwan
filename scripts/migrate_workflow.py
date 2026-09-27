"""Migrate Swwan IDs without assuming ownership of third-party name collisions."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT/'node_manifest.json').read_text())
ALIASES = {old:row['id'] for row in MANIFEST for old in row['legacy_ids']}
CURRENT = {row['id'] for row in MANIFEST}
SCALARS = {'INT','FLOAT','BOOLEAN','STRING','COLORCODE'}


def widget_fields(schema):
    return [(name,data) for group in ('required','optional') for name,data in schema.get(group,{}).items()
            if isinstance(data[0],list) or data[0] in SCALARS]


def migrate(document, selected=(), assume_swwan=False, registry=None):
    result=copy.deepcopy(document);selected={str(x) for x in selected};report={'changed':[], 'ambiguous':[]}
    wrapper=result.get('prompt') if isinstance(result,dict) else None
    if isinstance(wrapper,dict): entries=wrapper
    elif 'nodes' not in result:entries=result
    else:entries=None
    if entries is not None:
        for node_id,node in entries.items():
            if not isinstance(node,dict) or 'class_type' not in node:continue
            old=node['class_type'];meta=node.get('_meta',{})
            owned=assume_swwan or str(node_id) in selected or meta.get('swwan_version') is not None
            if old in ALIASES:
                if not owned:report['ambiguous'].append(str(node_id));continue
                node['class_type']=ALIASES[old];report['changed'].append(str(node_id))
            if node.get('class_type')=='ImageResizeKJv2Alternative' and node.get('inputs',{}).get('keep_proportion') is False:
                node['inputs']['keep_proportion']='stretch'
            if node.get('class_type') in CURRENT and (owned or old not in ALIASES):
                (node.setdefault('_meta',{}))['swwan_version']='1.0.0'
        return result,report
    for node in result['nodes']:
        old=node['type'];props=node.get('properties',{})
        owned=assume_swwan or str(node['id']) in selected or props.get('swwan_version') is not None or props.get('cnr_id') in {'comfyui_swwan','ComfyUI_Swwan'}
        if old in ALIASES and not owned:
            report['ambiguous'].append(str(node['id']));continue
        new=ALIASES.get(old,old)
        if new not in CURRENT:continue
        if old in ALIASES:node['type']=new;report['changed'].append(str(node['id']))
        if registry:
            cls=registry.NODE_CLASS_MAPPINGS[new]
            fields=widget_fields(cls.INPUT_TYPES());values=list(node.get('widgets_values') or [])
            # All extensions append widgets. Preserve converted / extra saved values.
            if len(values)<len(fields):
                for name,data in fields[len(values):]:
                    opts=data[1] if len(data)>1 else {}
                    values.append(opts.get('default',data[0][0] if isinstance(data[0],list) else None))
            for index,(name,_) in enumerate(fields):
                if new=='ImageResizeKJv2Alternative' and name=='keep_proportion' and index<len(values) and values[index] is False:
                    values[index]='stretch'
                if name=='font' and index<len(values) and values[index]=='TTNorms-Black.otf':values[index]='FreeMono.ttf'
            node['widgets_values']=values
            outputs=node.setdefault('outputs',[])
            for index,kind in enumerate(cls.RETURN_TYPES):
                if index>=len(outputs):outputs.append({'name':getattr(cls,'RETURN_NAMES',cls.RETURN_TYPES)[index],'type':kind,'links':None})
        props=node.setdefault('properties',{})
        props.update({'swwan_version':'1.0.0','cnr_id':'comfyui_swwan','Node name for S&R':new})
        for key in ('aux_id','ver'):props.pop(key,None)
    return result,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--swwan-node-id',action='append',default=[])
    parser.add_argument('--assume-swwan',action='store_true',help='Explicitly claim every legacy collision belongs to Swwan.')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--comfyui-root',type=Path)
    args=parser.parse_args();registry=None
    if args.comfyui_root:
        from migrate_qwen2511_workflow import load_nodes
        registry=load_nodes(args.comfyui_root)
    output=args.output or args.input.with_name(args.input.stem+'-swwan'+args.input.suffix)
    result,report=migrate(json.loads(args.input.read_text()),args.swwan_node_id,args.assume_swwan,registry)
    print(json.dumps(report,ensure_ascii=False))
    if not args.dry_run:
        if output.resolve()==args.input.resolve():parser.error('Original workflow must be preserved.')
        if report['ambiguous']:parser.error('Specify Swwan node IDs for ambiguous collisions; no output written.')
        with output.open('x',encoding='utf-8') as file:json.dump(result,file,ensure_ascii=False,indent=2);file.write('\n')
