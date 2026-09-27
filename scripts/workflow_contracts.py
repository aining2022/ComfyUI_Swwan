"""Strict, model-free validation of Swwan UI workflow contracts."""
import math
import re

SCALAR_TYPES = {'INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COLORCODE'}


def scalar_fields(cls):
    return {name: data for group in ('required', 'optional')
            for name, data in cls.INPUT_TYPES().get(group, {}).items()
            if isinstance(data[0], list) or data[0] in SCALAR_TYPES}


def validate_values(node, cls, values):
    fields = scalar_fields(cls)
    for name, value in values.items():
        if name not in fields:
            raise ValueError(f"Node #{node['id']}: unknown saved field {name}")
        kind, options = fields[name][0], fields[name][1] if len(fields[name]) > 1 else {}
        valid = True
        if isinstance(kind, list): valid = value in kind
        elif kind in ('INT', 'FLOAT'):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if kind == 'INT': valid = valid and isinstance(value, int)
            if valid: valid = options.get('min', -math.inf) <= value <= options.get('max', math.inf)
        elif kind == 'BOOLEAN': valid = isinstance(value, bool)
        elif kind in ('STRING', 'COLORCODE'): valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Node #{node['id']} {node['type']}: invalid {name}={value!r} for {kind!r}")
    # COLORCODE widgets persist hexadecimal literals; linked color objects are
    # handled by the graph and are never interpreted as saved widget values.
    for name, value in values.items():
        if fields[name][0] == 'COLORCODE' and not re.fullmatch(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{5})?', value):
            raise ValueError(f"Node #{node['id']}: invalid color {name}={value!r}")


def validate_workflow(workflow, registry):
    nodes = {n['id']: n for n in workflow['nodes']}
    if len(nodes) != len(workflow['nodes']): raise ValueError('Duplicate node IDs')
    links = {l[0]: l for l in workflow['links']}
    if len(links) != len(workflow['links']): raise ValueError('Duplicate link IDs')
    for lid, source, slot, target, index, kind in workflow['links']:
        try: out, inp = nodes[source]['outputs'][slot], nodes[target]['inputs'][index]
        except (KeyError, IndexError) as exc: raise ValueError(f'Invalid link {lid}') from exc
        if lid not in (out.get('links') or []) or inp.get('link') != lid:
            raise ValueError(f'Inconsistent link records {lid}')
        if kind != '*' and any(port['type'] not in ('*', kind) for port in (out, inp)):
            raise ValueError(f'Incompatible link {lid}')
    for node in workflow['nodes']:
        for inp in node.get('inputs', []):
            if inp.get('link') is not None and inp['link'] not in links: raise ValueError(f"Dangling input #{node['id']}")
        for out in node.get('outputs', []):
            if any(lid not in links for lid in out.get('links') or []): raise ValueError(f"Dangling output #{node['id']}")
        cls = registry.NODE_CLASS_MAPPINGS.get(node['type'])
        if cls is None: continue
        values = node.get('widgets_values_named')
        if not isinstance(values, dict): raise ValueError(f"Node #{node['id']}: missing named parameters")
        validate_values(node, cls, values)
        fields = scalar_fields(cls)
        inputs = {i['name']: i for i in node.get('inputs', [])}
        for name, data in cls.INPUT_TYPES().get('required', {}).items():
            if name in fields:
                if name not in values and inputs.get(name, {}).get('link') is None:
                    raise ValueError(f"Node #{node['id']}: missing required parameter {name}")
            elif inputs.get(name, {}).get('link') is None:
                raise ValueError(f"Node #{node['id']}: missing required input {name}")
    return workflow
