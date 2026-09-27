"""Repair the supplied face/head UI graph and retain the final SaveImage dependency chain.

This is a scoped repair, not an optimizer for arbitrary workflows. Set/Get bindings
are resolved without evaluating switches. No models are loaded or executed.
"""
import argparse
import copy
import json
from pathlib import Path
from migrate_qwen2511_workflow import defaults, load_nodes
from workflow_contracts import scalar_fields, validate_values, validate_workflow

ROOT = Path(__file__).resolve().parents[1]


def rebuild_links(workflow):
    nodes = {n['id']: n for n in workflow['nodes']}
    for node in nodes.values():
        for inp in node.get('inputs', []):
            if 'link' in inp: inp['link'] = None
        for out in node.get('outputs', []): out['links'] = None
    for lid, source, slot, target, index, _ in workflow['links']:
        out = nodes[source]['outputs'][slot]
        out['links'] = (out['links'] or []) + [lid]
        nodes[target]['inputs'][index]['link'] = lid


def repair(original, registry):
    result = copy.deepcopy(original)
    nodes = {n['id']: n for n in result['nodes']}
    expected = {62:'SwwanDrawMaskOnImage',92:'ImageResizeKJv2Alternative',
                47:'SwwanImageMatte',58:'SwwanImageMatte',162:'easy forLoopEnd',163:'SaveImage'}
    for nid, kind in expected.items():
        if nodes.get(nid, {}).get('type') != kind: raise ValueError(f'Expected {kind} #{nid}')
    reference = {n['id']:n for n in json.loads((ROOT/'examples/qwen-face-head-swwan.json').read_text())['nodes']}
    draw = nodes[62]
    if draw.get('widgets_values_named', {}).get('opacity') == 'cpu' and draw.get('widgets_values') == ['255, 255, 255','cpu','cpu']:
        draw['widgets_values_named'] = {'color':'255, 255, 255','opacity':1.0,'device':'cpu'}
    resize = nodes[92]
    if resize.get('widgets_values_named', {}).get('aspect_ratio') == '#364254':
        w = resize.get('widgets_values', [])
        ref = reference[92]['widgets_values_named']
        # Only the known missing-COLORCODE signature is recoverable. Lost final
        # interpolation is backed by the frozen original, never an inferred default.
        if len(w) != 24 or w[13:22] != ['#364254','original',1,1,'letterbox','lanczos','8','longest',1024]:
            raise ValueError('Resize #92: cannot reliably recover shifted fields')
        fields = list(defaults(registry.NODE_CLASS_MAPPINGS[resize['type']]))
        resize['widgets_values_named'] = dict(zip(fields, w + [ref['essentials_interpolation']]))
    for nid in (47,58):
        params = nodes[nid].setdefault('widgets_values_named', {})
        if 'background_color' not in params:
            params['background_color'] = reference[nid]['widgets_values_named']['background_color']
    # Drop loop display state only. Do not compact or reorder its optional ports.
    ignored_links = {i['link'] for i in nodes[162]['inputs']
                     if i['name'] in ('initial_value2','initial_value3') and i.get('link') is not None}
    result['links'] = [l for l in result['links'] if l[0] not in ignored_links]
    cache = nodes.get(165)
    if cache is not None:
        if cache['type'] != 'CachePreviewBridge' or cache.get('mode') != 4 or cache.get('widgets_values', [])[-1:] != [False]:
            raise ValueError('Cache #165 must be bypassed with caching disabled')
        incoming = [l for l in result['links'] if l[3] == 165 and cache['inputs'][l[4]]['name'] == 'images']
        if len(incoming) != 1: raise ValueError('Cache #165: missing unique image source')
        source = incoming[0]
        for link in result['links']:
            if link[1] == 165:
                if link[2] != 0: raise ValueError('Cache #165: connected non-image output')
                link[1:3] = source[1:3]
        result['links'] = [l for l in result['links'] if l[3] != 165]
        result['nodes'] = [n for n in result['nodes'] if n['id'] != 165]
        nodes.pop(165)
    setters = {}
    for node in nodes.values():
        if node['type'] == 'SetNode':
            name = node['widgets_values'][0]
            if name in setters: raise ValueError(f'Ambiguous virtual variable {name}')
            setters[name] = node['id']
    incoming = {}
    for link in result['links']: incoming.setdefault(link[3], []).append(link[1])
    keep = set()
    def visit(nid):
        if nid in keep: return
        keep.add(nid)
        node = nodes[nid]
        if node['type'] == 'GetNode':
            name = node['widgets_values'][0]
            if name not in setters: raise ValueError(f'Unresolved virtual variable {name}')
            visit(setters[name])
        for source in incoming.get(nid, []): visit(source)
    visit(163)
    result['nodes'] = [n for n in result['nodes'] if n['id'] in keep]
    result['links'] = [l for l in result['links'] if l[1] in keep and l[3] in keep]
    rebuild_links(result)
    for node in result['nodes']:
        cls = registry.NODE_CLASS_MAPPINGS.get(node['type'])
        if cls is None: continue
        values = node.get('widgets_values_named')
        if not isinstance(values, dict): raise ValueError(f"Node #{node['id']}: missing named values")
        validate_values(node, cls, values)
        fields = scalar_fields(cls)
        for name in cls.INPUT_TYPES().get('required', {}):
            if name in fields and name not in values and not any(i['name'] == name and i.get('link') is not None for i in node.get('inputs', [])):
                raise ValueError(f"Node #{node['id']}: missing required parameter {name}")
        merged = defaults(cls); merged.update(values)
        node['widgets_values_named'] = merged
        node['widgets_values'] = list(merged.values())
    # Only known link-reference metadata is pruned; layout coordinates and user
    # configuration are not interpreted as node/link IDs.
    links = {l[0] for l in result['links']}
    extra = result.get('extra', {})
    for key in ('links_added_by_ue','ue_links'):
        if extra.get(key): raise ValueError(f'Unsupported nonempty {key}; explicit resolution required')
    for reroute in extra.get('reroutes', []): reroute['linkIds'] = [lid for lid in reroute.get('linkIds', []) if lid in links]
    if 'reroutes' in extra: extra['reroutes'] = [r for r in extra['reroutes'] if r['linkIds']]
    if 'linkExtensions' in extra: extra['linkExtensions'] = [e for e in extra['linkExtensions'] if e['id'] in links]
    # Group rectangles contain no node references. Keep layout, remove settings
    # belonging solely to the deleted ignore-groups panel.
    if 'pixaromaGroups' in extra: extra['pixaromaGroups'] = []
    return validate_workflow(result, registry)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--comfyui-root', type=Path, required=True)
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists(): p.error('Choose a new output; existing files are never overwritten')
    result = repair(json.loads(args.input.read_text()), load_nodes(args.comfyui_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"Saved {args.output}: {len(result['nodes'])} nodes / {len(result['links'])} links")
