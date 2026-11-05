# SPDX-License-Identifier: MIT
"""Pruna task contracts and inspectable preprocessing; sampling stays in native nodes."""
import json
import math
import time

import torch
import torch.nn.functional as F
from .qwen21_geometry import (crop_area_resolution, normalized_mask, prepare_aligned,
                              restore_aligned)

TASKS = ('edit', 'outpaint', 'cutout', 'watermark', 'face_swap', 'photoreal', 'remove', 'angles')
LORAS = {
    'edit': '', 'outpaint': 'qwen21/Qwen2.1_扩图_GH_3750.safetensors',
    'cutout': 'qwen21/心宝强化抠图.safetensors',
    'watermark': 'qwen21/Qwen21-水印移除watermark_remover_v2.safetensors',
    'face_swap': 'qwen21/bfs_head_v1.1_qwen_2.1.safetensors',
    'photoreal': 'qwen21/pipi-qwen2.1-PhotoReal10.safetensors',
    'remove': 'qwen21/Qwen2.1_万物移除Remove_GH.safetensors',
    'angles': 'qwen21/angles_v2_full_qwen21_multiple_angles_v2_qwen21_multiple_angles_v2_000001500.safetensors',
}
STRENGTHS = dict(zip(TASKS, (0., 1., 1., .8, .7, .8, 1., .9)))
SIGMAS = '1.0, 0.961054, 0.918949, 0.769302, 0.545554, 0.222292, 0'
PROMPTS = {
    'edit': '', 'outpaint': '移除图像中的灰色区域，无缝自然延展为完整图像。',
    'cutout': 'Xinbao，抠出【红色标记出来的主体】，移除背景和所有外部遮挡物，补全其被遮挡的部分并显示完整主体；保持原有位置、比例、视角、颜色、材质和未遮挡细节不变，移除红色定位标记，输出透明背景PNG。',
    'watermark': 'remove all watermark.', 'photoreal': 'photorealistic,',
    'remove': '移除图像中的绿色区域，自然补全背景，保持其他内容不变。',
    'face_swap': 'head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving the hair, eye color, nose structure from <image2>. copy the direction of the eye and head rotation from <image1>; preserve the micro expressions from <image1>, high quality, sharp details, 4k.',
}


def is_text(config):
    return config['task'] == 'photoreal' and config['generation_mode'] == 'text'


def uses_references(config):
    return not is_text(config) and config['task'] in ('edit', 'face_swap')


def needs_mask(config):
    return config['task'] in ('remove', 'cutout', 'face_swap') or (
        config['task'] in ('edit', 'watermark') and config['edit_scope'] == 'local')


def resolve(values):
    config = dict(values)
    task = config['task']
    if task not in TASKS:
        raise ValueError('Unknown Pruna task')
    for name, choices in {
        'edit_scope': ('full', 'local'), 'generation_mode': ('image', 'text'),
        'encoder': ('auto', 'official', 'editutils'), 'color_mode': ('auto', 'none', 'goohai', 'bluebox'),
        'marking': ('brush', 'circle'), 'padding_unit': ('pixels', 'percent'),
        'additional_lora_mode': ('off', 'edit_only', 'all'),
    }.items():
        if config[name] not in choices:
            raise ValueError('Invalid ' + name)
    for key, value in config.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError('Non-finite ' + key)
    crop_area_resolution(1, 1, config['qwen_max_resolution'], config['qwen_min_resolution'])
    if not .2 <= config['qwen_tolerance'] <= 3 or not 1 <= config['context_factor'] <= 10:
        raise ValueError('Invalid geometry controls')
    if not 1 <= config['people'] <= 6 or not 0 <= config['color_strength'] <= 1:
        raise ValueError('Invalid people/color controls')
    if not 0 <= config['skin_strength'] <= 1 or not 0 <= config['additional_lora_strength'] <= 2:
        raise ValueError('Invalid skin/additional LoRA strength')
    if any(not 0 <= config[key] <= 30 for key in ('expand_percent','feather_percent')):
        raise ValueError('Invalid restoration controls')
    if config['lora_strength'] != -1 and not 0 <= config['lora_strength'] <= 2:
        raise ValueError('lora_strength must be -1 (task preset) or 0..2')
    for name in ('left', 'right', 'top', 'bottom', 'seam_band'):
        if not 0 <= config[name] <= 4096:
            raise ValueError('Invalid ' + name)
    if not 0 <= config['mark_opacity'] <= 1 or not 8 <= config['block_size'] <= 512:
        raise ValueError('Invalid marker controls')
    for name in ('output_width', 'output_height'):
        if config[name] < 32 or config[name] > 4096 or config[name] % 32:
            raise ValueError(name + ' must be a multiple of 32 in 32..4096')
    if task == 'edit' and not config['extra_prompt'].strip():
        raise ValueError('请输入具体编辑指令，尚未开始推理。')
    prompt = PROMPTS.get(task, '')
    if task == 'angles':
        from .qwen21_tasks import AZIMUTHS, ELEVATIONS
        if config['azimuth'] not in range(0, 360, 30) or config['elevation'] not in ELEVATIONS:
            raise ValueError('Unsupported camera angle')
        prompt = f"<mva> {AZIMUTHS[config['azimuth']//30]}, {ELEVATIONS[config['elevation']]}"
        if config['close_up']:
            prompt += ' close-up'
    config['prompt'] = '\n'.join(x for x in (prompt, config['extra_prompt'].strip()) if x)
    config['task_lora'] = config['task_lora_' + task].strip()
    config['active_lora_strength'] = STRENGTHS[task] if config['lora_strength'] == -1 else config['lora_strength']
    config['active_additional_strength'] = config['additional_lora_strength'] if (
        config['additional_lora_mode'] == 'all' or
        config['additional_lora_mode'] == 'edit_only' and task == 'edit') else 0.
    config['sigmas'] = SIGMAS
    config['schema_version'] = 1
    return config


def valid_image(image):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('Pruna workflow requires one RGB/RGBA image, not a batch')
    if not torch.isfinite(image).all():
        raise ValueError('Image contains non-finite values')
    return image


def load_red_mask(path):
    """Read mask samples without the video decoder's RGB range conversion."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        red = np.array(ImageOps.exif_transpose(source).convert('RGB').getchannel('R'), dtype=np.float32)
    return torch.from_numpy(red / 255.)[None]


def selection(image, mask, protection, config):
    if mask is None:
        raise ValueError('此任务需要非空选区 MASK；不要将主图 alpha 自动当作选区。')
    selected = normalized_mask(mask, image)
    if config['marking'] == 'circle':
        from scipy.ndimage import binary_fill_holes
        support = selected[0].detach().cpu().numpy() > 0
        filled = binary_fill_holes(support)
        if not (filled & ~support).any():
            raise ValueError('圈选未闭合，请补画闭合圈或选择 brush。')
        selected = torch.as_tensor(filled, device=image.device, dtype=torch.float32)[None]
    selected = (selected - protection).clamp(0, 1)
    if config['enhanced'] and config['task'] == 'remove':
        from ..workflow_tools import BlockifyMask
        selected = BlockifyMask().process(selected, config['block_size'], 'cpu')[0].to(image.device)
        selected = (selected - protection).clamp(0, 1)
    if not bool(torch.any(selected > 0)):
        raise ValueError('Selection has no selected pixels after protection')
    return selected


def prepare(image, mask, protection, config):
    start = time.perf_counter()
    if is_text(config):
        # This is diagnostic geometry only. The encoder creates a native 64-channel latent.
        w, h = config['output_width'], config['output_height']
        canvas = torch.zeros(1, h, w, 3)
        return canvas, torch.zeros(1, h, w), {'text': True, 'canvas_size': (w, h)}, json.dumps({'mode': 'text', 'canvas_size': [w, h]})
    image = valid_image(image)
    if config['task'] == 'face_swap':
        raise ValueError('BFS must use the dedicated active-only face branch')
    task = config['task']
    protected = torch.zeros_like(image[..., 0]) if protection is None else normalized_mask(protection, image, allow_empty=True)
    if task == 'cutout' and bool(torch.any(protected)):
        raise ValueError('Cutout protection is not an alpha contract; disconnect protect_mask')
    margins = None
    if task == 'outpaint':
        h, w = image.shape[1:3]
        values = [config[k] for k in ('left', 'right', 'top', 'bottom')]
        if config['padding_unit'] == 'percent':
            values = [int(size*v/100) for size, v in zip((w,w,h,h), values)]
        left, right, top, bottom = values
        if not any(values):
            raise ValueError('扩图至少需要一侧大于0像素')
        if (w+left+right)*(h+top+bottom) > 64*1024*1024:
            raise ValueError('Expanded canvas exceeds 64 megapixels; reduce margins')
        from .outpaint import directional_outpaint
        canvas, allowed = directional_outpaint(image[..., :3], left, top, right, bottom, '像素', 0, 0)
        if image.shape[-1] == 4:
            alpha = torch.ones_like(canvas[..., :1])
            alpha[:, top:top+h, left:left+w] = image[..., 3:4].to(alpha)
            canvas = torch.cat((canvas, alpha), -1)
        margins = values
        p = torch.zeros_like(allowed)
        p[:, top:top+h, left:left+w] = protected.cpu()
        band = min(config['seam_band'], w, h)
        if band:
            if left: allowed[:,top:top+h,left:left+band] = 1
            if right: allowed[:,top:top+h,left+w-band:left+w] = 1
            if top: allowed[:,top:top+band,left:left+w] = 1
            if bottom: allowed[:,top+h-band:top+h,left:left+w] = 1
        image, protected = canvas, p
        allowed = (allowed-protected).clamp(0,1)
    elif needs_mask(config):
        allowed = selection(image, mask, protected, config)
    else:
        allowed = (torch.ones_like(image[..., 0])-protected).clamp(0,1)
    # Full-image transforms use the whole canvas, even if protected pixels are present.
    full = task in ('angles', 'photoreal') or task in ('edit', 'watermark') and config['edit_scope'] == 'full'
    geometry_mask = torch.ones_like(allowed) if full else allowed
    if not bool(torch.any(allowed > 0)):
        raise ValueError('Selection has no selected pixels after protection')
    clean, selected, seam = prepare_aligned(image, geometry_mask,
        tolerance=config['qwen_tolerance'], factors=(config['context_factor'],)*4,
        resolution_mode='crop_area', min_resolution=config['qwen_min_resolution'], max_resolution=config['qwen_max_resolution'])
    seam['original_mask'] = allowed
    seam.update(pruna_version=1, task=task, protection=protected, prepared_clean=clean,
                prepared_mask=selected, margins=margins, strict_local=not full)
    report = dict(task=task, native_crop_size=seam['native_crop_size'], canvas_size=seam['canvas_size'],
                  working_resolution=seam['resolution'], prepare_s=time.perf_counter()-start,
                  selected_source_pixels=int((allowed>0).sum()), selected_work_pixels=int((selected>0).sum()))
    return clean, selected, seam, json.dumps(report, ensure_ascii=False)


def mark(image, mask, config):
    if config['task'] not in ('remove', 'cutout'):
        return image
    weight = mask
    color = (0., 1., 0.)
    opacity = config['mark_opacity']
    if config['task'] == 'cutout':
        support = (mask > 0).float()[:, None]
        # Interior outline, including the canvas boundary; never treat it as output alpha.
        erosion = -F.max_pool2d(-F.pad(support, (2,2,2,2)), 5, 1)
        weight = (support-erosion).clamp(0,1)[:,0]
        color, opacity = (1., 0., 0.), 1.
    elif config['enhanced']:
        opacity = 1.
    result = image.clone()
    alpha = weight[...,None].to(image)*opacity
    result[...,:3] = image[...,:3]*(1-alpha)+image.new_tensor(color)*alpha
    return result


def color_mode(config, seam):
    mode = config['color_mode']
    if config['task'] == 'cutout' or is_text(config):
        if mode not in ('auto','none'):
            raise ValueError('Cutout/text mode does not apply RGB background color restoration')
        return 'none'
    if mode != 'auto':
        return mode
    if config['task'] in ('remove','outpaint') or config['task'] in ('edit','watermark') and config['edit_scope']=='local':
        return 'goohai'
    return 'none'


def restore(image, seam, config):
    start = time.perf_counter()
    if seam.get('text'):
        if image.shape[1:3] != tuple(reversed(seam['canvas_size'])):
            raise ValueError('Text generation output dimensions differ from requested canvas')
        return image, torch.ones_like(image[...,0]), json.dumps({'task':config['task'],'generation_mode':'text'})
    if image.shape[1:3] != tuple(reversed(seam['canvas_size'])):
        raise ValueError('Decoded size differs from the single recorded geometry transform')
    mode = color_mode(config, seam)
    background = seam['original_image']
    allowed = seam['original_mask'].to(image.device)
    protected = seam['protection'].to(image.device)
    if config['task'] == 'cutout':
        result, fusion = restore_aligned(background, image, seam, transparent=True)
    elif mode == 'goohai':
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get('GH_CropRestore')
        if cls is None:
            raise RuntimeError('Install Goohaitools with GH_CropRestore color restoration, or select color_mode=none')
        from .qwen21_gh_seam import to_goohai_seam
        gh_seam = to_goohai_seam(seam)
        gh_seam['原始图像'] = background[..., :3]
        result = getattr(cls(), cls.FUNCTION)(gh_seam, image[..., :3],
            config['expand_percent'], config['feather_percent'], seam['prepared_mask'],
            颜色还原强度=config['color_strength'])[0]
        result = result.to(background)
        fusion = allowed.to(background)
    else:
        processed = image
        if mode == 'bluebox' and config['color_strength'] > 0:
            import nodes
            cls = nodes.NODE_CLASS_MAPPINGS.get('ColorCalibratorBlueBox')
            if cls is None:
                raise RuntimeError('Install BlueBox ColorCalibratorBlueBox or select color_mode=none')
            selected = seam['prepared_mask'] if seam['strict_local'] else None
            output = getattr(cls(), cls.FUNCTION)(seam['prepared_clean'][..., :3], image[..., :3],
                threshold=.08, expand=64, open_k=13, close_k=51, strength=config['color_strength'],
                outside_mode='edited', feather=24, mask=selected)
            processed = output['result'][0] if isinstance(output,dict) else output[0]
        result, fusion = restore_aligned(background, processed, seam,
            feather=round(min(seam['native_crop_size'])*config['feather_percent']/100) if seam['strict_local'] else 0,
            strict=True)
    if config['task'] != 'cutout':
        # Restore native alpha for edits; full-white protection is exact after every color/fusion path.
        if background.shape[-1] == 4:
            result = torch.cat((result[..., :3], background[...,3:4].to(result)), -1)
        keep = (allowed <= 0) | (protected >= 1)
        result = torch.where(keep[...,None].to(result.device), background.to(result), result)
        fusion = torch.where(keep.to(fusion.device), 0., fusion)
    report = dict(task=config['task'], color_backend=mode, output_size=list(result.shape[1:3]),
                  channels=result.shape[-1], restore_s=time.perf_counter()-start,
                  gpu_generation_verified=False)
    if config['task'] != 'cutout':
        report['outside_changed_values'] = int(torch.count_nonzero((result-background.to(result))[(allowed<=0).to(result.device)]))
    return result, fusion, json.dumps(report,ensure_ascii=False)
