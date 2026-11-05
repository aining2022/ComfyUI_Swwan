# SPDX-License-Identifier: MIT
"""Build a visible BFS loop from its source canvas, without task-wrapper nodes.

This module only transforms workflow data. No model, GPU or ComfyUI import is
required. The caller supplies the source UI graphs and an object_info schema.
"""
import copy
import hashlib
import json
import uuid


SCALARS = {'STRING', 'INT', 'FLOAT', 'BOOLEAN', 'COLORCODE', 'COMBO'}
SIGMAS = '1.0, 0.961054, 0.918949, 0.769302, 0.545554, 0.222292, 0'


def _named(node):
    values = node.get('widgets_values_named')
    if not isinstance(values, dict):
        raise ValueError(f"Source node {node['id']} needs named widget values")
    return values


def _set(node, **values):
    current = dict(_named(node))
    current.update(values)
    node['widgets_values_named'] = current
    node['widgets_values'] = list(current.values())


def _constant(node):
    named = node.get('widgets_values_named') or {}
    return named.get('Constant', (node.get('widgets_values') or [''])[0])


def _schema_node(schemas, kind, node_id, title, values=None):
    schema = schemas.get(kind)
    if schema is None:
        raise ValueError(f'Missing object_info schema for {kind}')
    values = values or {}
    saved, inputs = {}, []
    for group in ('required', 'optional'):
        for name, declaration in schema['input'].get(group, {}).items():
            field = declaration[0]
            options = declaration[1] if len(declaration) > 1 else {}
            scalar = isinstance(field, list) or field in SCALARS
            if scalar and not options.get('forceInput'):
                default = options.get('default')
                if default is None:
                    choices = field if isinstance(field, list) else options.get('options')
                    default = choices[0] if choices else {'INT': 0, 'FLOAT': 0., 'BOOLEAN': False}.get(field, '')
                saved[name] = values.get(name, default)
                if options.get('control_after_generate'):
                    saved['control_after_generate'] = 'fixed'
            else:
                inputs.append({'name': name, 'type': field, 'link': None,
                               **({'shape': 7} if group == 'optional' else {})})
    unknown = set(values) - set(saved)
    if unknown:
        raise ValueError(f'{kind}: unknown saved fields {sorted(unknown)}')
    names = schema.get('output_name') or schema['output']
    return {'id': node_id, 'type': kind, 'title': title, 'pos': [0, 0],
            'size': [390, 180], 'flags': {}, 'order': 0, 'mode': 0,
            'inputs': inputs, 'outputs': [{'name': name, 'type': field, 'links': []}
                                         for name, field in zip(names, schema['output'])],
            'properties': {'Node name for S&R': kind},
            'widgets_values': list(saved.values()), 'widgets_values_named': saved}


def _edges(workflow):
    nodes = {n['id']: n for n in workflow['nodes']}
    if len(nodes) != len(workflow['nodes']):
        raise ValueError('Duplicate BFS source IDs')
    edges = {}
    for _, source, output, target, slot, _ in workflow['links']:
        key = target, nodes[target]['inputs'][slot]['name']
        if key in edges:
            raise ValueError(f'Duplicate source connection: {key}')
        edges[key] = source, output
    return nodes, edges


def _resolve_virtual_wires(nodes, edges):
    setters = {}
    for node in nodes.values():
        if node['type'] == 'SetNode':
            name = _constant(node)
            if name in setters:
                raise ValueError(f'Duplicate SetNode name: {name}')
            linked = [edges[(node['id'], item['name'])] for item in node.get('inputs', [])
                      if (node['id'], item['name']) in edges]
            if len(linked) != 1:
                raise ValueError(f'SetNode {name} must have one source')
            setters[name] = linked[0]

    def resolve(edge, seen=()):
        source, slot = edge
        if source in seen:
            raise ValueError('Cyclic virtual wiring')
        node = nodes[source]
        if node['type'] == 'GetNode':
            name = _constant(node)
            if name not in setters:
                raise ValueError(f'Missing SetNode for {name}')
            return resolve(setters[name], (*seen, source))
        if node['type'] == 'SetNode':
            return resolve(setters[_constant(node)], (*seen, source))
        return source, slot

    for key, edge in list(edges.items()):
        edges[key] = resolve(edge)
    for node_id in [k for k, n in nodes.items() if n['type'] in ('SetNode', 'GetNode')]:
        nodes.pop(node_id)
        for key in [key for key in edges if key[0] == node_id]:
            edges.pop(key)


def _serialize_links(nodes, edges):
    links = []
    for node in nodes.values():
        for item in node.get('inputs', []):
            item['link'] = None
        for item in node.get('outputs', []):
            item['links'] = []
    for (target, name), (source, slot) in sorted(edges.items()):
        if source not in nodes or target not in nodes:
            raise ValueError(f'Missing endpoint: {source} -> {target}.{name}')
        output = nodes[source]['outputs'][slot]
        target_node = nodes[target]
        fields = target_node.setdefault('inputs', [])
        index = next((i for i, item in enumerate(fields) if item['name'] == name), None)
        if index is None:
            fields.append({'name': name, 'type': output['type'], 'link': None,
                           'widget': {'name': name}})
            index = len(fields) - 1
        kind = output['type']
        accepted = fields[index]['type']
        if kind not in ('*', accepted) and accepted not in ('*', 'ANY') and kind != 'ANY':
            raise ValueError(f'Type mismatch {source}:{kind} -> {target}.{name}:{accepted}')
        link = [len(links) + 1, source, slot, target, index, kind]
        fields[index]['link'] = link[0]
        output['links'].append(link[0])
        links.append(link)
    return links


def _layout(nodes):
    groups = []
    buckets = [
        ('01 / 原图与六张有序参考；不是任务封装',
         [140, 200, 170, 148, 149, 150, 151, 283, 284], 0, 0, 3),
        ('02 / 明示控制：人数、换头、额头、肤色与恢复',
         [225, 189, 194, 195, 272, 273, 274, 275, 191, 271, 91010], 0, 2200, 3),
        ('03 / 原图检测与排序；固定原图区域，不重检生成图',
         [1, 2, 7, 35, 74, 77, 78, 86, 88, 89, 90, 93, 94, 102, 104,
          117, 122, 124, 129, 157, 158, 159, 213, 224, 226, 233, 234, 230, 232,
          101, 208], 1550, 0, 4),
        ('04 / 可见逐人循环与同源面积预算裁剪',
         [72, 110, 73, 82, 83, 130, 133, 214, 215, 216, 217, 218, 219, 131,
          91021, 91022, 91023, 91024], 3650, 0, 3),
        ('05 / 参考脸对齐、白底与编码图；实际尺寸可见',
         [202, 211, 12, 13, 91, 92, 287, 288, 160, 161, 80, 53, 56, 59, 61,
          67, 68, 155, 156, 282, 91025], 5200, 0, 4),
        ('06 / 原生 EditUtils 配置、编码、Pruna 六步采样',
         [235, 91001, 237, 91002, 236, 238, 239, 242, 250, 241, 246, 91003,
          91004, 91005, 91006, 247, 248, 249, 91026], 7300, 0, 4),
        ('07 / 显式肤色匹配与混合；默认关闭',
         [220, 153, 154, 47, 58, 60, 66, 70, 285], 9400, 0, 3),
        ('08 / 同一 SEAM 恢复、循环结束、原图保护与保存',
         [63, 162, 91020, 163, 221, 222, 223, 276], 10950, 0, 3),
    ]
    placed = set()
    for title, ids, x, y, columns in buckets:
        active = [nodes[i] for i in ids if i in nodes and i not in placed]
        top = y + 80
        for offset in range(0, len(active), columns):
            row = active[offset:offset+columns]
            heights = []
            for col, node in enumerate(row):
                named_count = len(node.get('widgets_values_named') or {})
                height = max(170, 80 + 24 * named_count + 16 * len(node.get('inputs', [])))
                if node['type'] in ('LoadImage', 'PreviewImage', 'SwwanImageAndMaskPreview', 'SaveImage'):
                    height = max(380, height)
                node['pos'] = [x + 25 + col * 500, top]
                node['size'] = [460, height]
                node['flags'] = {}
                heights.append(height)
                placed.add(node['id'])
            top += max(heights, default=0) + 80
        groups.append({'title': title, 'bounding': [x, y, columns*500+20, top-y],
                       'color': '#38655e', 'font_size': 24, 'flags': {}})
    # Retain any source diagnostic not listed above, without silently hiding it.
    rest = [n for n in nodes.values() if n['id'] not in placed]
    top = 2900
    for offset in range(0, len(rest), 3):
        row = rest[offset:offset+3]
        for col, node in enumerate(row):
            height = 960 if node['type'] == 'MarkdownNote' else 390
            node['pos'] = [10975 + col*500, top]
            node['size'] = [460, height]
        top += max(node['size'][1] for node in row) + 80
    if rest:
        groups.append({'title': '09 / 补充说明与原始可见检查点',
                       'bounding': [10950, 2820, 1520, top-2820],
                       'color': '#655c38', 'font_size': 24, 'flags': {}})
    for i, group in enumerate(groups):
        group['id'] = i + 1
    return groups


def build_bfs(source, pruna_source, schemas):
    """Return an independent, fully visible BFS UI canvas; never mutate inputs."""
    workflow = copy.deepcopy(source)
    nodes, edges = _edges(workflow)
    original = {n['id']: copy.deepcopy(n) for n in pruna_source['nodes']}
    for required in (63, 72, 131, 162, 202, 235, 236, 237, 238, 241, 242, 250):
        if required not in nodes:
            raise ValueError(f'Not the expected BFS v5 canvas: missing {required}')
    for required in (509, 527, 548, 511, 532, 522, 523, 524, 525, 526, 562, 566, 575):
        if required not in original:
            raise ValueError(f'Not the expected Pruna v3 canvas: missing {required}')
    _resolve_virtual_wires(nodes, edges)

    def remove(node_id):
        nodes.pop(node_id, None)
        for key in [key for key in edges if key[0] == node_id]:
            edges.pop(key)

    def clone_core(old_id, new_id, title):
        item = copy.deepcopy(original[old_id])
        item.update(id=new_id, title=title, mode=0, flags={})
        nodes[new_id] = item
        return item

    def add(kind, node_id, title, **values):
        nodes[node_id] = _schema_node(schemas, kind, node_id, title, values)
        return nodes[node_id]

    for target, slot in list(edges):
        edge = edges[(target, slot)]
        # The old post-crop long-edge resize must not introduce another transform.
        replacements = {(138, 0): (131, 0), (138, 3): (131, 5)}
        edges[(target, slot)] = replacements.get(edge, edge)
    for node_id in (138, 280, 281, 240, 243, 245, 260, 261, 262, 263, 264, 265, 266, 267):
        remove(node_id)

    clone_core(509, 235, '原 v3 INT8 DiT / 不使用 Turbo DiT')
    clone_core(548, 236, '原 v3 INT8 CLIP')
    clone_core(511, 238, '原 v3 Qwen 2.1 VAE')
    clone_core(527, 91001, 'Pruna v0.2 六步 LoRA / 权重1')
    clone_core(532, 91002, '原 v3 Attention Backend / 不用 BlockCache 或 Spectrum')
    clone_core(524, 91003, '可见随机种子 / 固定对照；每人沿用同一seed')
    clone_core(526, 91004, 'CFG Guider / 原 v3 CFG=1')
    clone_core(523, 91005, 'KSampler Select / Euler')
    clone_core(522, 91006, '原 v3 完整 Pruna 六步 sigmas')
    clone_core(525, 247, 'Sampler Custom Advanced / 逐人六步')
    clone_core(562, 91010, 'BFS触发提示词 / 原 v3 原文，直接可编辑')
    if _named(nodes[91006])['sigmas'] != SIGMAS:
        raise ValueError('Unexpected Pruna sigma schedule')
    _set(nodes[91003], control_after_generate='fixed')
    _set(nodes[237], lora_name=_named(original[566])['lora_5_name'],
         strength_model=_named(original[575])['value'])
    _set(nodes[189], **{'值': _named(original[575])['value']})
    nodes[237]['title'] = 'BFS v1.1任务LoRA / 路径与权重直接显示'
    edges.pop((237, 'lora_name'), None)
    nodes[237]['inputs'] = [i for i in nodes[237]['inputs'] if i['name'] != 'lora_name']
    edges.update({(91001, 'model'): (235, 0), (237, 'model'): (91001, 0),
                  (91002, 'model'): (237, 0), (91004, 'model'): (91002, 0),
                  (91004, 'positive'): (241, 0), (91004, 'negative'): (246, 0),
                  (247, 'noise'): (91003, 0), (247, 'guider'): (91004, 0),
                  (247, 'sampler'): (91005, 0), (247, 'sigmas'): (91006, 0),
                  (247, 'latent_image'): (241, 1), (241, 'prompt'): (91010, 0)})
    for name in ('model', 'positive', 'negative'):
        edges.pop((247, name), None)

    _set(nodes[131], crop_mode='qwen21_aligned', fill_mask_holes=False,
         qwen_resolution_mode='crop_area', qwen_min_resolution=128,
         qwen_max_resolution=1024, qwen_tolerance=1., qwen_restore_backend='swwan')
    nodes[131]['title'] = '主图局部裁剪 / BlueBox / 面积预算128..1024 / 唯一SEAM'
    nodes[191]['title'] = '辅助分割裁剪最小长边 / 非Qwen面积预算'
    nodes[271]['title'] = '辅助分割裁剪最大长边 / 非Qwen面积预算'
    nodes[91]['title'] = '辅助检测裁剪 / 保留原BFS上下文，不送Qwen'
    nodes[92]['title'] = '辅助分割图缩放 / 不是主图第二次缩放'
    add('SwwanMaskCombine', 91021, '当前人物原坐标mask减保护 / 空选区由裁剪拒绝',
        operation='相减', bbox_mode='关闭', alignment='居中')
    mask_fields = [name for group in ('required', 'optional')
                   for name, field in schemas['SwwanMaskCombine']['input'].get(group, {}).items()
                   if field[0] == 'MASK']
    if len(mask_fields) != 2:
        raise ValueError('Expected two Mask Combine inputs')
    edges[(91021, mask_fields[0])] = (73, 0)
    edges[(91021, mask_fields[1])] = (284, 0)
    edges[(131, 'mask')] = (91021, 0)
    edges[(91, 'mask')] = (91021, 0)
    nodes[284]['title'] = '原图尺寸保护mask / 默认全黑，可接精确文件mask替换'
    edges.pop((63, 'protect_mask'), None)
    nodes[63]['inputs'] = [i for i in nodes[63]['inputs'] if i['name'] != 'protect_mask']
    nodes[63]['title'] = '同源SEAM恢复 / 原图BOX必连 / 不重复缩放'
    add('ImageCompositeMasked', 91020, '最终保护 / 白色从原图取回，灰色软保护',
        x=0, y=0, resize_source=False)
    edges.update({(91020, 'destination'): (162, 0), (91020, 'source'): (140, 0),
                  (91020, 'mask'): (284, 0), (163, 'images'): (91020, 0)})

    # The main image is already aligned. Give EditUtils its actual longest edge.
    edges[(242, 'image')] = (131, 0)
    edges[(282, 'image')] = (131, 0)
    edges[(242, 'ref_longest_edge')] = (282, 0)
    _set(nodes[242], ref_resize_mode='longest_edge', ref_crop='pad',
         to_ref=True, ref_main_image=True, to_vl=True, rope_x_offset=0, rope_y_offset=0)
    nodes[242]['title'] = '主图EditUtils配置 / 使用实际长边，保持已对齐画布'
    add('easy imageSizeByLongerSide', 91025, '参考图自身长边 / 不套用主图裁剪尺寸')
    edges[(91025, 'image')] = (13, 0)
    edges[(250, 'ref_longest_edge')] = (91025, 0)
    nodes[250]['title'] = '参考图EditUtils配置 / 独立长边与独立身份图'
    nodes[249]['title'] = '仅移除编码器padding / 后续由同一SEAM恢复原坐标'
    # References are selected lazily; source-canvas bypassed loaders would make
    # selecting people 2..6 fail even when their filenames are supplied.
    for node_id in (200, 170, 148, 149, 150, 151):
        nodes[node_id]['mode'] = 0
    for node in nodes.values():
        schema = schemas.get(node['type'], {})
        allowed = {name for group in ('required', 'optional')
                   for name in schema.get('input', {}).get(group, {})}
        named = node.get('widgets_values_named')
        if isinstance(named, dict):
            node['widgets_values_named'] = {name: value for name, value in named.items()
                                           if name in allowed or name in ('upload', 'control_after_generate')}
            node['widgets_values'] = list(node['widgets_values_named'].values())
    _set(nodes[163], filename_prefix='Qwen21_Pruna_v5_Expanded/BFS')
    for node_id, slot, title in [(91022, 9, '预算依据 / 原生裁剪宽'),
                                (91023, 10, '预算依据 / 原生裁剪高'),
                                (91024, 11, '实际面积预算 R / 不是长边')]:
        add('PreviewAny', node_id, title)
        edges[(node_id, 'source')] = (131, slot)
    add('PreviewImage', 91026, '原始VAE解码 / 未去padding、未校色、未贴回')
    edges[(91026, 'images')] = (248, 0)

    note = (
        '## BFS 展开版 / 手动检查入口\n\n'
        '没有 Pruna Face/Prepare/Encode/Restore 任务包装；没有子图或 Set/Get 隐形接线。'
        'easy forLoopStart/End 直接显示在画布上，内部节点按每人执行。\n\n'
        '1. 原图和参考1..6逐个加载；人物顺序仍沿用源BFS左右上下排序。'
        '选区默认来自原图 LoadImage 的 MASK，表示涂绘或1-alpha，不是自动主体mask；'
        '普通不透明图为空mask时由检测链自动检测。\n'
        '2. 保护默认全黑。若要文件保护，使用精确文件读取→ImageToMask(red)，'
        '替换保护SolidMask输出；选择和保护须与原图同尺寸。\n'
        '3. 原图检测与区域排序在循环外，逐人参考用同一index。'
        '参考图槽位必须至少填写处理人数，不自动替代缺失身份。\n'
        '4. 主裁剪131面积预算128..1024；辅助分割91/92保留原BFS长边范围，'
        '它不送Qwen。主裁剪后不再接独立Resize。\n'
        '5. 保留原v3 INT8模型、Pruna权重1、BFS v1.1权重0.7、原触发词、CFG1、Euler和完整6步sigmas；'
        '不启用BlockCache/Spectrum。\n'
        '6. 本版是展开后的检查稿，静态接线检查不代表GPU效果验收。'
        '未声称保留所有模型内部语义位置，也未验证GPU常驻LoRA。'
        '原BFS参考对齐/分割/肤色链保留可见，恢复采用当前aligned SEAM的严格范围语义。'
    )
    nodes[91030] = {'id': 91030, 'type': 'MarkdownNote', 'title': 'BFS检查说明与未验边界',
                   'pos': [0, 0], 'size': [460, 700], 'flags': {}, 'order': 0, 'mode': 0,
                   'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [note]}
    for node_id in [k for k, n in nodes.items() if n['type'] in ('Note', 'MarkdownNote') and k != 91030]:
        remove(node_id)
    for item in nodes.values():
        if item['type'].startswith('SwwanQwen21'):
            raise ValueError('A hidden Qwen task wrapper remained')
    links = _serialize_links(nodes, edges)
    groups = _layout(nodes)
    for order, item in enumerate(nodes.values()):
        item['order'] = order
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    workflow = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'swwan-pruna-v5-bfs:' + fingerprint)),
                'revision': 0, 'last_node_id': max(nodes), 'last_link_id': len(links),
                'nodes': list(nodes.values()), 'links': links, 'groups': groups,
                'config': {}, 'extra': {'ds': {'scale': .25, 'offset': [50, 50]},
                    'swwan_expanded_version': 5, 'source_bfs_sha256': fingerprint,
                    'validation': 'static graph only; expanded BFS GPU run pending'}, 'version': .4}
    validate_bfs(workflow)
    return workflow


def to_api(workflow, schemas):
    """Export actual schema fields; Easy-Use owns its visible loop expansion."""
    validate_bfs(workflow)
    nodes = {n['id']: n for n in workflow['nodes']}
    links = {edge[0]: edge for edge in workflow['links']}
    result = {}
    for node in workflow['nodes']:
        if node['type'] in ('Note', 'MarkdownNote') or node.get('mode', 0) != 0:
            continue
        schema = schemas[node['type']]
        allowed = {name for group in ('required', 'optional')
                   for name in schema['input'].get(group, {})}
        inputs = {name: value for name, value in (node.get('widgets_values_named') or {}).items()
                  if name in allowed}
        for item in node.get('inputs', []):
            if item.get('link') is None:
                continue
            if item['name'] not in allowed:
                raise ValueError(f"Unknown API input {node['id']}.{item['name']}")
            _, source, slot, _, _, _ = links[item['link']]
            if nodes[source].get('mode', 0) != 0:
                raise ValueError('Active API input depends on a disabled source')
            inputs[item['name']] = [str(source), slot]
        required = schema['input'].get('required', {})
        if set(required) - set(inputs):
            raise ValueError(f"Missing API inputs on {node['id']}: {set(required) - set(inputs)}")
        result[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs,
                                  '_meta': {'title': node.get('title', node['type'])}}
    return result


def validate_bfs(workflow):
    """Check explicit graph reciprocity, no hidden wrappers, and core contracts."""
    nodes = {n['id']: n for n in workflow['nodes']}
    if len(nodes) != len(workflow['nodes']):
        raise ValueError('Duplicate node IDs')
    if workflow.get('definitions'):
        raise ValueError('Hidden subgraph definitions are not allowed')
    links = {edge[0]: edge for edge in workflow['links']}
    if len(links) != len(workflow['links']):
        raise ValueError('Duplicate link IDs')
    for node in nodes.values():
        if node['type'].startswith('SwwanQwen21') or node['type'] in ('SetNode', 'GetNode'):
            raise ValueError(f"Hidden routing remains: {node['type']}")
        for index, item in enumerate(node.get('inputs', [])):
            if item.get('link') is not None:
                link = links.get(item['link'])
                if link is None or link[3:5] != [node['id'], index]:
                    raise ValueError('Dangling or mismatched input')
        for slot, item in enumerate(node.get('outputs', [])):
            for link_id in item.get('links') or []:
                link = links.get(link_id)
                if link is None or link[1:3] != [node['id'], slot]:
                    raise ValueError('Dangling or mismatched output')
    for link_id, source, slot, target, index, _ in links.values():
        if link_id not in nodes[source]['outputs'][slot]['links']:
            raise ValueError('Output link mismatch')
        if nodes[target]['inputs'][index]['link'] != link_id:
            raise ValueError('Input link mismatch')
    if _named(nodes[91006])['sigmas'] != SIGMAS or _named(nodes[91004])['cfg'] != 1:
        raise ValueError('Pruna sampling changed')
    if _named(nodes[131])['qwen_min_resolution'] != 128 or _named(nodes[131])['qwen_max_resolution'] != 1024:
        raise ValueError('Unexpected main-crop budget')
    if nodes[72]['type'] != 'easy forLoopStart' or nodes[162]['type'] != 'easy forLoopEnd':
        raise ValueError('BFS loop is not visible')
    incoming = {(target, nodes[target]['inputs'][index]['name']): (source, slot)
                for _, source, slot, target, index, _ in links.values()}
    expected = {(72, 'initial_value1'): (226, 0), (226, 'in'): (140, 0),
                (131, 'image'): (72, 2), (63, 'background_image'): (72, 2),
                (63, 'crop_box'): (131, 2), (63, 'region_info'): (131, 4),
                (162, 'flow'): (72, 0), (162, 'initial_value1'): (63, 0),
                (110, 'start_index'): (72, 1), (202, 'index'): (72, 1)}
    if any(incoming.get(key) != value for key, value in expected.items()):
        raise ValueError('BFS loop, reference ordering, or shared SEAM changed')

    ancestors, active = {}, set()

    def visit(node_id):
        if node_id in active:
            raise ValueError('Cyclic BFS graph')
        if node_id not in ancestors:
            active.add(node_id)
            result = set()
            for (target, _), (source, _) in incoming.items():
                if target == node_id:
                    result.add(source)
                    result.update(visit(source))
            active.remove(node_id)
            ancestors[node_id] = result
        return ancestors[node_id]

    for node_id in nodes:
        visit(node_id)
    if any(ancestors[node_id] & {72, 162} for node_id in (124, 233, 208)):
        raise ValueError('Person detection/order must come from the original image')
    return workflow
