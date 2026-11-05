# SPDX-License-Identifier: MIT
"""Adapt registered Qwen 2.1 encoders without hiding resizes or changing backends."""
import math

import torch
import torch.nn.functional as F

from .qwen21_geometry import resize_image


def text_only(config):
    return config.get('task') == 'photoreal' and config.get('generation_mode', 'image') == 'text'


def _registry():
    import nodes
    return nodes.NODE_CLASS_MAPPINGS


def _invoke(registry, node_id, **inputs):
    cls = registry.get(node_id)
    if cls is None:
        raise RuntimeError(f'Required node {node_id} is unavailable; install/update its provider. No encoder fallback was applied.')
    method = 'execute' if callable(getattr(cls, 'define_schema', None)) else getattr(cls, 'FUNCTION', 'execute')
    output = getattr(cls(), method)(**inputs)
    if isinstance(output, dict):
        output = output.get('result')
    elif hasattr(output, 'result'):
        output = output.result
    if not isinstance(output, (tuple, list)):
        raise RuntimeError(f'{node_id} returned an unsupported output contract')
    return output


def _image(image, name, aligned=False):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f'{name} must contain exactly one RGB or RGBA image')
    h, w = image.shape[1:3]
    if min(h, w) < 1 or (aligned and (h % 32 or w % 32)):
        raise ValueError(f'{name} must have positive dimensions aligned to 32 before encoding')
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f'{name} contains non-finite pixels')
    return image


def prepare_references(references, resolution):
    if not isinstance(references, (list, tuple)):
        raise ValueError('References must be an ordered image list')
    if len(references) > 15:
        raise ValueError('Qwen 2.1 accepts at most 15 auxiliary references after the main image')
    if not isinstance(resolution, int) or isinstance(resolution, bool) or not 32 <= resolution <= 2048 or resolution % 32:
        raise ValueError('Reference resolution must be a multiple of 32 in 32..2048')
    prepared = []
    for index, source in enumerate(references):
        source = _image(source, f'Reference {index + 1}')
        h, w = source.shape[1:3]
        scale = min(1., resolution / math.sqrt(h * w))
        width, height = max(1, round(w * scale)), max(1, round(h * scale))
        image = resize_image(source, width, height)
        right, bottom = (-width) % 32, (-height) % 32
        image = F.pad(image.movedim(-1, 1), (0, right, 0, bottom), mode='replicate').movedim(1, -1)
        prepared.append(image)
    return prepared


def _canvas(config):
    width, height = config.get('output_width', 1024), config.get('output_height', 1024)
    for value in (width, height):
        if not isinstance(value, int) or isinstance(value, bool) or not 32 <= value <= 4096 or value % 32:
            raise ValueError('Text canvas width/height must be multiples of 32 in 32..4096')
    return width, height


def _latent(latent, width, height):
    if not isinstance(latent, dict) or not torch.is_tensor(latent.get('samples')):
        raise RuntimeError('Qwen encoder did not return a latent samples tensor')
    expected = (1, 64, height // 16, width // 16)
    if tuple(latent['samples'].shape) != expected:
        raise RuntimeError(f'Qwen 2.1 latent shape {tuple(latent["samples"].shape)} differs from {expected}; refusing a shifted or incompatible canvas')


def _record(backend, width, height, images, **extra):
    return dict(version=1, backend=backend, canvas_size=(width, height),
                content_size=(width, height), image_sizes=[(x.shape[2], x.shape[1]) for x in images],
                pad_info={'x': 0, 'y': 0, 'width': 0, 'height': 0, 'scale_by': 1.}, **extra)


def encode(config, clip, vae, image=None, references=None, registry=None):
    registry = _registry() if registry is None else registry
    backend = config.get('encoder', 'auto')
    if backend not in ('auto', 'official', 'editutils'):
        raise ValueError('Encoder must be auto, official or editutils')
    is_text = text_only(config)
    if backend == 'auto':
        backend = 'editutils' if config.get('task') in ('cutout', 'face_swap') else 'official'
    prompt = config.get('prompt', '')
    negative_prompt = config.get('negative_prompt', '')
    if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
        raise ValueError('Prompt and negative prompt must be text')
    if backend == 'editutils' and negative_prompt.strip():
        raise ValueError('EditUtils compatibility uses ConditioningZeroOut; clear negative_prompt or explicitly select official encoding')
    if is_text:
        if backend != 'official':
            raise ValueError('Pure text generation requires the official Qwen 2.1 encoder')
        width, height = _canvas(config)
        positive, negative, latent = _invoke(registry, 'TextEncodeQwenImage21', clip=clip, vae=vae,
                                             prompt=prompt, negative_prompt=negative_prompt, resolution=0, images={})
        # Official Qwen 2.1 uses 64 channels at 16x downscale. No VAE image/reference is encoded.
        _latent(latent, 1024, 1024)
        latent = dict(latent, samples=latent['samples'].new_zeros((1, 64, height // 16, width // 16)))
        return positive, negative, latent, _record(backend, width, height, [], text_only=True)

    image = _image(image, 'Main image', aligned=True)
    references = prepare_references(references if references is not None else [],
                                    config.get('reference_resolution', 768))
    if config.get('task') == 'face_swap' and len(references) != 1:
        raise ValueError('Each BFS invocation requires exactly one aligned identity reference')
    if config.get('task') == 'cutout' and references:
        raise ValueError('The cutout compatibility branch accepts only its marked main image')
    images = [image, *references]
    height, width = image.shape[1:3]
    if backend == 'official':
        positive, negative, latent = _invoke(
            registry, 'TextEncodeQwenImage21', clip=clip, vae=vae, prompt=prompt,
            negative_prompt=negative_prompt, resolution=0,
            images={f'image_{index + 1}': item for index, item in enumerate(images)})
        _latent(latent, width, height)
        return positive, negative, latent, _record(backend, width, height, images, text_only=False)

    model_config = _invoke(registry, 'QwenImage21ModelConfig_EditUtils', instruction='')[0]
    configs = []
    for index, item in enumerate(images):
        configs.append(dict(image=item, to_ref=not (config.get('task') == 'cutout' and index == 0),
                            ref_main_image=index == 0, to_vl=True, ref_longest_edge=max(item.shape[1:3]),
                            ref_crop='pad', ref_upscale='lanczos', ref_resize_mode='longest_edge',
                            rope_x_offset=0, rope_y_offset=0))
    output = _invoke(registry, 'EditTextEncode_EditUtils', clip=clip, vae=vae, prompt=prompt,
                     model_config=model_config, configs=configs)
    if len(output) != 6:
        raise RuntimeError('EditUtils must expose conditioning/latent/custom/main_image/mask/pad_info')
    positive, latent, custom, main_image, _, pad_info = output
    _image(main_image, 'EditUtils returned main image', aligned=True)
    if tuple(main_image.shape[1:3]) != (height, width):
        raise RuntimeError('EditUtils resized the prepared main image; refusing a second geometry transform')
    _latent(latent, width, height)
    if not isinstance(pad_info, dict):
        raise RuntimeError('EditUtils did not return padding metadata')
    if any(pad_info.get(key, 0) != 0 for key in ('x', 'y', 'width', 'height')) or pad_info.get('scale_by', 1.) != 1.:
        raise RuntimeError('EditUtils added unexpected padding/scaling to an already aligned main image')
    negative = _invoke(registry, 'ConditioningZeroOut', conditioning=positive)[0]
    count = len(custom.get('ref_latents', [])) if isinstance(custom, dict) else None
    padding = _record(backend, width, height, images, text_only=False,
                      requested_to_ref=[item['to_ref'] for item in configs], actual_reference_count=count)
    # Recent EditUtils forces VL-only Qwen 2.1 inputs into reference slots; expose that fact.
    padding['cutout_reference_promoted'] = config.get('task') == 'cutout' and count is not None and count > 0
    return positive, negative, latent, padding


def unpad(image, padding):
    image = _image(image, 'Decoded image')
    if not isinstance(padding, dict) or padding.get('version') != 1:
        raise ValueError('Expected padding from the same Pruna encode node')
    width, height = padding['canvas_size']
    if tuple(image.shape[1:3]) != (height, width):
        raise ValueError('Decoded canvas differs from the encoder canvas')
    # Both supported encoders are required to preserve the prepared canvas. Keep native alpha.
    return image
