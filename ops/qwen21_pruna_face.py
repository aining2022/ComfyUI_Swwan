# SPDX-License-Identifier: MIT
"""Reuse frozen BFS preprocessing with a shared, externally patched Pruna model."""
import json
import math
import time

import torch

from .qwen21_geometry import normalized_mask, restore_aligned
from .qwen21_graph import dependencies, face_graph

PRUNA_SIGMAS = '1.0, 0.961054, 0.918949, 0.769302, 0.545554, 0.222292, 0'


def validate_config(config):
    if config.get('task') != 'face_swap':
        raise ValueError('Pruna Face only accepts task=face_swap')
    people = config.get('people', 1)
    if isinstance(people, bool) or int(people) != people or not 1 <= people <= 6:
        raise ValueError('Pruna Face people must be an integer in 1..6')
    for name in ('expand_percent', 'feather_percent'):
        value = float(config.get(name, 5))
        if not math.isfinite(value) or not 0 <= value <= 30:
            raise ValueError(f'{name} must be in 0..30')
    strength = float(config.get('skin_strength', .25))
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('skin_strength must be in 0..1')


def source_masks(image, mask=None, protect_mask=None):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('Pruna Face requires one RGB or RGBA image, not an image batch')
    if not bool(torch.isfinite(image).all()):
        raise ValueError('Pruna Face image contains non-finite values')
    protection = torch.zeros_like(image[..., 0]) if protect_mask is None else normalized_mask(
        protect_mask, image, allow_empty=True)
    selection = torch.zeros_like(image[..., 0]) if mask is None else normalized_mask(
        mask, image, allow_empty=True)
    manual = bool(torch.any(selection > 0))
    selection = (selection - protection).clamp(0, 1)
    if manual and not bool(torch.any(selection > 0)):
        raise ValueError('The manual face selection is fully protected; no sampling started')
    if bool(torch.all(protection >= 1)):
        raise ValueError('The image is fully protected; no sampling started')
    return selection, protection, manual


def guarded_selection(image, mask, protect_mask):
    selection = normalized_mask(mask, image)
    protection = normalized_mask(protect_mask, image, allow_empty=True)
    selected = (selection - protection).clamp(0, 1)
    if not bool(torch.any(selected > 0)):
        raise ValueError('Face region has no editable pixels after protection; no sampling started')
    return selected


def _node(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def person_graph(model, clip, vae, image, mask, protect_mask, references, config, manual=False):
    validate_config(config)
    legacy = dict(config, resolution=1024, people=config.get('people', 1),
                  swap_head=config.get('swap_head', False), forehead=config.get('forehead', False),
                  skin_match=config.get('skin_match', False))
    graph = face_graph(model, clip, vae, image, mask, references, legacy)
    graph['226'] = _node('SwwanQwen21PrunaFaceGuard', image=['140', 0], count=['117', 0],
                        valid=['224', 7], references=references, config=config)
    graph['reference'] = _node('SwwanQwen21PrunaFaceReferences', image=['13', 0])
    graph['effective'] = _node('SwwanQwen21PrunaFaceMask', image=['72', 2],
                              mask=['73', 0], protect_mask=protect_mask)
    graph['131']['inputs'].update(
        image=['72', 2], mask=['effective', 0], fill_mask_holes=False,
        crop_mode='qwen21_aligned', qwen_resolution_mode='crop_area',
        qwen_min_resolution=config.get('qwen_min_resolution', 128),
        qwen_max_resolution=config.get('qwen_max_resolution', 1024),
        qwen_tolerance=config.get('qwen_tolerance', 1.), qwen_restore_backend='swwan')
    graph['encode'] = _node('SwwanQwen21PrunaEncode', clip=clip, vae=vae, config=config,
                           image=['131', 0], references=['reference', 0])
    graph['noise'] = _node('RandomNoise', noise_seed=config['seed'])
    graph['guider'] = _node('CFGGuider', model=model, positive=['encode', 0], negative=['encode', 1], cfg=1.)
    graph['sampler'] = _node('KSamplerSelect', sampler_name='euler')
    graph['sigmas'] = _node('ManualSigmas', sigmas=PRUNA_SIGMAS)
    graph['sample'] = _node('SamplerCustomAdvanced', noise=['noise', 0], guider=['guider', 0],
                           sampler=['sampler', 0], sigmas=['sigmas', 0], latent_image=['encode', 2])
    graph['decode'] = _node('VAEDecode', samples=['sample', 0], vae=vae)
    graph['247'] = _node('SwwanQwen21PrunaUnpad', image=['decode', 0], padding=['encode', 3])
    graph['63'] = _node('SwwanQwen21PrunaFaceRestore', background=['72', 2], original=image,
                       image=['285', 0], seam=['131', 4], mask=['133', 0], manual=manual,
                       selection=mask, protect_mask=protect_mask, config=config, person=['72', 1], previous_report='')
    graph['final'] = _node('SwwanQwen21PrunaFaceFinish', image=['162', 0], original=image,
                          protect_mask=protect_mask, report=['63', 1], count=config.get('people', 1), config=config)
    graph['273']['inputs']['值'] = config.get('skin_strength', .25)
    # A supplied selection remains manual even when smaller than the legacy area threshold.
    graph['7']['inputs'].update(boolean=manual, on_true=mask)
    for item in graph.values():
        for key, value in list(item['inputs'].items()):
            if isinstance(value, list) and len(value) == 2 and value == ['129', 7]:
                item['inputs'][key] = manual
    graph.pop('checkpoint', None)
    return graph


def build_face_graph(model, clip, vae, image, mask, protect_mask, references, config, manual=False):
    from comfy_execution.graph_utils import GraphBuilder, add_graph_prefix
    graph = person_graph(model, clip, vae, image, mask, protect_mask, references, config, manual)
    graph['sequence'] = _node('SwwanQwen21PrunaFaceSequence', model=model, clip=clip, vae=vae,
                             image=['226', 0], mask=mask, protect_mask=protect_mask,
                             regions=['233', 0], manual=manual, count=['117', 0],
                             references=references, config=config)
    graph = dependencies(graph, ['sequence'])
    graph, outputs = add_graph_prefix(graph, (['sequence', 0], ['sequence', 1]), GraphBuilder.alloc_prefix())
    return {'result': outputs, 'expand': graph}


def build_face_sequence(model, clip, vae, image, mask, protect_mask, regions, manual, count, references, config):
    from comfy_execution.graph_utils import GraphBuilder, add_graph_prefix
    validate_config(config)
    if not 1 <= count <= min(6, config.get('people', 1), len(regions[1])):
        raise ValueError('Invalid frozen face-region count; no sampling started')
    if len(references) < count:
        raise ValueError(f'{count} face regions require {count} references')
    expanded = {}
    previous, report = image, ''
    for index in range(count):
        graph = person_graph(model, clip, vae, image, mask, protect_mask, references, config, manual)
        constants = {('72', 1): index, ('72', 2): image, ('124', 0): regions,
                     ('233', 0): regions, ('208', 0): 0.}
        for item in graph.values():
            for key, value in list(item['inputs'].items()):
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    item['inputs'][key] = constants.get(tuple(value), value)
        graph = dependencies(graph, ['63'])
        prefix = GraphBuilder.alloc_prefix()
        graph, outputs = add_graph_prefix(graph, (['63', 0], ['63', 1]), prefix)
        # Previous results are external links and must not receive this person's prefix.
        for node_id, field in (('131', 'image'), ('effective', 'image'), ('63', 'background'), ('91', 'image')):
            if prefix + node_id in graph:
                graph[prefix + node_id]['inputs'][field] = previous
        graph[prefix + '63']['inputs']['previous_report'] = report
        expanded.update(graph)
        previous, report = outputs
    graph = GraphBuilder()
    done = graph.node('SwwanQwen21PrunaFaceFinish', image=previous, original=image,
                      protect_mask=protect_mask, report=report, count=count, config=config)
    expanded.update(graph.finalize())
    return {'result': (done.out(0), done.out(1)), 'expand': expanded}


def restore_person(background, original, image, seam, mask, manual, selection,
                   protect_mask, config, person, previous_report=''):
    validate_config(config)
    started = time.perf_counter()
    if image.shape[1:3] != (seam['canvas_size'][1], seam['canvas_size'][0]):
        raise ValueError('BFS output dimensions differ from the recorded Pruna canvas')
    selection = normalized_mask(selection, original, allow_empty=True)
    protection = normalized_mask(protect_mask, original, allow_empty=True)
    if background.shape != original.shape:
        raise ValueError('BFS original and sequential canvas shapes differ')
    x0, y0, x1, y1 = seam['box']
    side = min(x1-x0, y1-y0)
    result, fusion = restore_aligned(
        background, image, seam, cropped_mask=mask, protect_mask=protection,
        expand=round(side * config.get('expand_percent', 5) / 100),
        feather=round(side * config.get('feather_percent', 5) / 100), strict=manual)
    if manual:
        gate = selection > 0
        result = torch.where(gate[..., None], result, background)
        fusion = torch.where(gate, fusion, 0)
    result = torch.where((protection >= 1)[..., None], original, result)
    fusion = torch.where(protection >= 1, 0, fusion)
    prior = json.loads(previous_report) if previous_report else {'people_records': []}
    records = list(prior['people_records'])
    if len(records) != person:
        raise ValueError('BFS person order/report is inconsistent; refusing to mix runs')
    records.append(dict(person=person+1, status='person_completed',
                        canvas_size=list(seam['canvas_size']), native_crop_size=list(seam['native_crop_size']),
                        working_resolution=seam['resolution'], manual_mask=manual,
                        restore_s=time.perf_counter()-started,
                        protected_changed_values=int(torch.count_nonzero((result-original)[protection >= 1]))))
    return result, json.dumps({'people_records': records}, ensure_ascii=False)


def finish(image, original, protect_mask, report, count, config):
    validate_config(config)
    if image.shape != original.shape:
        raise ValueError('BFS final image dimensions/channels differ from the original')
    records = json.loads(report)['people_records']
    if len(records) != count or any(r['status'] != 'person_completed' for r in records):
        raise ValueError('Incomplete BFS task')
    protection = normalized_mask(protect_mask, original, allow_empty=True)
    image = torch.where((protection >= 1)[..., None], original, image)
    result = dict(task='face_swap', status='completed', sampling='pruna_v0.2_6step',
                  people_processed=count, people_records=records,
                  output_size=list(image.shape[1:3]), gpu_residency_verified=False,
                  protection_changed_values=int(torch.count_nonzero((image-original)[protection >= 1])))
    return image, json.dumps(result, ensure_ascii=False)
