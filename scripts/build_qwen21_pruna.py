# SPDX-License-Identifier: MIT
"""Build inspectable Pruna v4 UI/API workflows from the user's v3, without overwrites."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid

from migrate_qwen2511_workflow import defaults, load_nodes
from workflow_contracts import validate_workflow

CORE_IDS = (509, 527, 548, 511, 532, 522, 523, 524, 526, 525, 513)
IDS = dict(controls=600, inputs=601, model=602, prepare=603, mark=604, encode=605,
           unpad=606, restore=607, face=608, output=609, save=610, note=611,
           prompt_preview=612, image_preview=613)
SCALARS = {'INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COLORCODE'}


def _new_node(registry, kind, node_id, title, position, values=None, size=(380, 220)):
    cls = registry.NODE_CLASS_MAPPINGS[kind]
    settings = defaults(cls)
    for name, value in (values or {}).items():
        if name not in settings:
            raise ValueError(f'{kind} has no scalar field {name}')
        settings[name] = value
    sockets = []
    for group in ('required', 'optional'):
        for name, spec in cls.INPUT_TYPES().get(group, {}).items():
            if isinstance(spec[0], list) or spec[0] in SCALARS:
                continue
            sockets.append({'name': name, 'type': spec[0], 'link': None,
                            **({'shape': 7} if group == 'optional' else {})})
    return {'id': node_id, 'type': kind, 'title': title, 'pos': list(position),
            'size': list(size), 'flags': {}, 'order': 0, 'mode': 0, 'inputs': sockets,
            'outputs': [{'name': name, 'type': kind, 'links': []}
                        for name, kind in zip(getattr(cls, 'RETURN_NAMES', cls.RETURN_TYPES), cls.RETURN_TYPES)],
            'properties': {'Node name for S&R': kind}, 'widgets_values': list(settings.values()),
            'widgets_values_named': settings}


def to_api(workflow):
    """Export named values plus actual links; never zip widget positions into API names."""
    links = {link[0]: link for link in workflow['links']}
    graph = {}
    for node in workflow['nodes']:
        if node.get('mode', 0) != 0 or node['type'] in ('MarkdownNote', 'Note'):
            continue
        inputs = dict(node.get('widgets_values_named') or {})
        inputs.pop('control_after_generate', None)
        for socket in node.get('inputs', []):
            if socket.get('link') is not None:
                _, source, slot, _, _, _ = links[socket['link']]
                inputs[socket['name']] = [str(source), slot]
        graph[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs,
                                  '_meta': {'title': node.get('title', node['type'])}}
    return graph


def build(original, registry):
    source = {n['id']: n for n in original['nodes']}
    for node_id in (*CORE_IDS, 530, 551, 555, 566):
        if node_id not in source:
            raise ValueError(f'Not the expected Pruna v3 source: missing node {node_id}')
    nodes = {}
    positions = {509: (0, 100), 527: (0, 245), 548: (0, 410), 511: (0, 580),
                 532: (0, 935), 522: (1880, 310), 523: (1880, 445), 524: (1880, 565),
                 526: (1880, 110), 525: (2260, 240), 513: (2600, 240)}
    for node_id in CORE_IDS:
        node = copy.deepcopy(source[node_id])
        node.update(pos=list(positions[node_id]), mode=0)
        for socket in node.get('inputs', []): socket['link'] = None
        for socket in node.get('outputs', []): socket['links'] = []
        nodes[node_id] = node
    # One seed belongs to the task config; native randomization cannot fork BFS's seed.
    nodes[524]['widgets_values_named']['control_after_generate'] = 'fixed'
    if len(nodes[524].get('widgets_values', [])) > 1:
        nodes[524]['widgets_values'][1] = 'fixed'
    nodes[524]['title'] = '种子 / 与 BFS 共用；在任务控制中修改'

    selected = source[566]['widgets_values_named']
    tasks = ('edit', 'outpaint', 'cutout', 'watermark', 'face_swap', 'photoreal', 'remove', 'angles')
    control_values = dict(task='edit', extra_prompt=source[555]['widgets_values_named']['value'],
                          seed=source[524]['widgets_values_named']['noise_seed'],
                          additional_lora_mode='edit_only',
                          additional_lora_name=source[530]['widgets_values_named']['lora_name'],
                          additional_lora_strength=source[530]['widgets_values_named']['strength_model'])
    control_values.update({f'task_lora_{task}': selected[f'lora_{index+1}_name']
                           for index, task in enumerate(tasks)})
    specifications = [
        ('controls', 'SwwanQwen21PrunaControls', '01 / 任务、预算、恢复与附加 LoRA',
         (610, 100), control_values, (540, 1640)),
        ('inputs', 'SwwanQwen21PrunaInputs', '02 / 文件名或外接输入；选区遮罩独立于 alpha',
         (1210, 100), {'image_filename': source[551]['widgets_values_named']['image'],
                       'mask_filename': '', 'protect_filename': '', 'reference_filenames': ''}, (520, 430)),
        ('model', 'SwwanQwen21PrunaModel', '任务 LoRA / 干净 Pruna 底模，不串未激活 LoRA',
         (0, 735), None, (510, 130)),
        ('prepare', 'SwwanQwen21PrunaPrepare', '03 / 裁剪、上下文、面积预算与唯一几何记录',
         (1210, 610), None, (520, 220)),
        ('mark', 'SwwanQwen21PrunaMark', '04 / 副本任务标记；保留干净图作校色参考',
         (1210, 920), None, (520, 150)),
        ('encode', 'SwwanQwen21PrunaEncode', '05 / 官方或 EditUtils 兼容编码',
         (1210, 1180), None, (520, 250)),
        ('unpad', 'SwwanQwen21PrunaUnpad', '07 / 验证编码画布；拒绝未记录的二次缩放',
         (2600, 430), None, (390, 150)),
        ('restore', 'SwwanQwen21PrunaRestore', '08 / 校色、几何恢复、保护或原生 RGBA',
         (3040, 240), None, (450, 220)),
        ('face', 'SwwanQwen21PrunaFace', 'BFS 专用 / 固定人物序、参考图与逐人共享采样',
         (1880, 1000), None, (600, 300)),
        ('output', 'SwwanQwen21PrunaOutput', '09 / 惰性输出路由；仅执行选中任务',
         (3540, 240), None, (420, 180)),
        ('save', 'SwwanSaveImage', '10 / 最终 RGB / RGBA PNG；保留生成 alpha',
         (4020, 240), {'file_format': 'png', 'filename_prefix': 'Qwen21_Pruna_v4',
                       'input_mode': 'straight', 'alpha_mode': 'auto', 'save_workflow_as_json': True}, (470, 600)),
    ]
    for key, kind, title, pos, values, size in specifications:
        nodes[IDS[key]] = _new_node(registry, kind, IDS[key], title, pos, values, size)
    nodes[IDS['note']] = {
        'id': IDS['note'], 'type': 'MarkdownNote', 'title': '开始使用 / 关键输入合同',
        'pos': [0, 1150], 'size': [510, 650], 'flags': {}, 'order': 0, 'mode': 0,
        'inputs': [], 'outputs': [], 'properties': {},
        'widgets_values': ['## Pruna v4 多任务整合\n\n'
            '1. 在任务控制中选择 task；输入填写当前 ComfyUI input 目录中的文件名。\n'
            '2. 局部编辑、移除、抠图需独立的白色选区 mask。LoadImage MASK 是反向 alpha，不是自动选区。\n'
            '3. 参考图文件名每行一个；BFS 按原图固定人物顺序匹配，支持 1..6 人。\n'
            '4. photoreal + text 不读取图片、选区、保护或参考。\n'
            '5. 128..1024 是面积平方根预算，不是长边。低于 512 的模型质量尚未 GPU 验收。\n'
            '6. auto 抠图/BFS 需要 EditUtils；auto 局部编辑/水印/移除/扩图需要 GH 校色。\n'
            '7. 中间图片预览默认禁用；BFS 时启用它会执行非选中的通用分支。\n'
            '8. 保留 v3 模型/CLIP/VAE 和 Pruna 六步参数；未实现 GPU 常驻适配器切换。\n\n'
            '详细中文说明：docs/QWEN21_PRUNA.md'],
    }
    # A lightweight prompt preview does not force image preprocessing or an inactive sampler.
    prompt_preview = copy.deepcopy(source[518])
    prompt_preview.update(id=IDS['prompt_preview'], title='实际送入模型的任务提示词', mode=0,
                          pos=[610, 1850], size=[540, 230])
    for socket in prompt_preview.get('inputs', []): socket['link'] = None
    for socket in prompt_preview.get('outputs', []): socket['links'] = []
    nodes[prompt_preview['id']] = prompt_preview
    nodes[IDS['image_preview']] = {
        'id': IDS['image_preview'], 'type': 'PreviewImage', 'title': '可选模型输入预览 / 默认禁用，避免执行非选分支',
        'pos': [1210, 1530], 'size': [520, 400], 'flags': {}, 'order': 0, 'mode': 2,
        'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': None}], 'outputs': [],
        'properties': {'Node name for S&R': 'PreviewImage'}, 'widgets_values': [], 'widgets_values_named': {},
    }
    links = []
    def connect(source_id, source_slot, target_id, name):
        target = nodes[target_id]
        if not any(i['name'] == name for i in target.get('inputs', [])):
            kind = nodes[source_id]['outputs'][source_slot]['type']
            target.setdefault('inputs', []).append({'name': name, 'type': kind, 'link': None,
                                                   'widget': {'name': name}})
        slot = next(i for i, item in enumerate(target['inputs']) if item['name'] == name)
        socket = target['inputs'][slot]
        if socket.get('link') is not None: raise ValueError(f'Duplicate connection {target_id}.{name}')
        kind = nodes[source_id]['outputs'][source_slot]['type']
        if socket['type'] != '*' and kind != socket['type']:
            raise ValueError(f'Wrong type {source_id}:{source_slot} -> {target_id}.{name}')
        link_id = len(links) + 1
        links.append([link_id, source_id, source_slot, target_id, slot, kind])
        socket['link'] = link_id
        nodes[source_id]['outputs'][source_slot].setdefault('links', []).append(link_id)
    c, inp, model, prep, mark, enc, unpad, restore, face, output, save = (
        IDS[name] for name in ('controls', 'inputs', 'model', 'prepare', 'mark', 'encode',
                              'unpad', 'restore', 'face', 'output', 'save'))
    connect(509, 0, 527, 'model'); connect(527, 0, model, 'model'); connect(model, 0, 532, 'model')
    for target in (inp, model, prep, mark, enc, restore, face, output): connect(c, 0, target, 'config')
    connect(c, 1, IDS['prompt_preview'], 'source'); connect(c, 2, 524, 'noise_seed')
    for target in (prep, face):
        connect(inp, 0, target, 'image'); connect(inp, 1, target, 'mask'); connect(inp, 2, target, 'protect_mask')
    connect(inp, 3, enc, 'references'); connect(inp, 3, face, 'references')
    connect(prep, 0, mark, 'image'); connect(prep, 1, mark, 'mask')
    connect(mark, 0, enc, 'image'); connect(mark, 0, IDS['image_preview'], 'images')
    connect(548, 0, enc, 'clip'); connect(511, 0, enc, 'vae')
    connect(532, 0, 526, 'model'); connect(enc, 0, 526, 'positive'); connect(enc, 1, 526, 'negative')
    connect(524, 0, 525, 'noise'); connect(526, 0, 525, 'guider'); connect(523, 0, 525, 'sampler')
    connect(522, 0, 525, 'sigmas'); connect(enc, 2, 525, 'latent_image')
    connect(525, 0, 513, 'samples'); connect(511, 0, 513, 'vae')
    connect(513, 0, unpad, 'image'); connect(enc, 3, unpad, 'padding')
    connect(unpad, 0, restore, 'image'); connect(prep, 2, restore, 'seam')
    connect(532, 0, face, 'model'); connect(548, 0, face, 'clip'); connect(511, 0, face, 'vae')
    connect(restore, 0, output, 'generic'); connect(face, 0, output, 'face')
    connect(output, 0, save, 'image')
    for order, node in enumerate(nodes.values()): node['order'] = order
    fingerprint = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    groups = [
        ('共享 v3 模型 / Pruna / 当前任务 LoRA', [-40, 30, 590, 1070]),
        ('任务控制与实际提示词', [570, 30, 600, 2090]),
        ('输入 / 几何 / 标记 / 编码', [1170, 30, 610, 1950]),
        ('共享 Pruna 六步采样核心', [1840, 30, 1170, 750]),
        ('BFS 专用：惰性逐人处理', [1840, 930, 700, 460]),
        ('恢复 / 选中输出 / RGBA 安全保存', [3000, 140, 1540, 800]),
    ]
    workflow = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'swwan-pruna-v4:' + fingerprint)),
                'revision': 0, 'last_node_id': max(nodes), 'last_link_id': len(links),
                'nodes': list(nodes.values()), 'links': links,
                'groups': [{'id': i+1, 'title': title, 'bounding': bounds, 'color': '#3f789e',
                            'font_size': 24, 'flags': {}} for i, (title, bounds) in enumerate(groups)],
                'config': {}, 'extra': {'ds': {'scale': .5, 'offset': [70, 40]},
                    'swwan_pruna_version': 4, 'source_semantic_sha256': fingerprint}, 'version': .4}
    return validate_workflow(workflow, registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--api-output', type=Path)
    parser.add_argument('--baseline-output', type=Path)
    parser.add_argument('--comfyui-root', type=Path, required=True)
    options = parser.parse_args()
    outputs = [p for p in (options.output, options.api_output, options.baseline_output) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs): parser.error('Output paths must be distinct')
    for path in outputs:
        if path.resolve() == options.source.resolve() or path.exists():
            parser.error(f'Output must be new and distinct from the source: {path}')
    source_bytes = options.source.read_bytes()
    workflow = build(json.loads(source_bytes), load_nodes(options.comfyui_root))
    for path in outputs: path.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + '\n')
    if options.api_output:
        options.api_output.write_text(json.dumps(to_api(workflow), ensure_ascii=False, indent=2) + '\n')
    if options.baseline_output: options.baseline_output.write_bytes(source_bytes)
    print(f'Created {options.output}: {len(workflow["nodes"])} nodes, {len(workflow["links"])} links')


if __name__ == '__main__':
    main()
