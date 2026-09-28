# SPDX-License-Identifier: GPL-3.0-only
# Directional geometry, gray canvas and Gaussian feathering adapted from
# goohai/Goohaitools-comfyui, nodes/外补画板（3合一）.py at
# a84303e6e73a289af59d96eddab9f521ebd00643. Copyright goohai and contributors.
# Swwan modifications: explicit tensor-mask normalization, validation, and
# an optional branch of the existing outpaint node; no plugin imports.
"""Directional outpainting in pixels or percentages of each original axis."""
import numpy as np
import torch
from PIL import Image, ImageFilter


def directional_outpaint(image, left, top, right, bottom, unit, alignment, feathering, mask=None):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError('Directional outpaint expects one RGB image; use legacy mode for image batches.')
    image = image.detach().cpu()
    _, height, width, _ = image.shape
    crop_only = left == right == top == bottom == 0 and alignment > 0
    sides = [left, right, top, bottom]
    if unit == '百分比':
        sides = [int(size * value / 100) for size, value in zip([width, width, height, height], sides)]
    left, right, top, bottom = sides
    target_width, target_height = width + left + right, height + top + bottom
    if alignment > 0:
        for size, start, end, axis in [(target_width, left, right, 'width'), (target_height, top, bottom, 'height')]:
            aligned = (size // alignment if crop_only else (size + alignment - 1) // alignment) * alignment
            delta = aligned - size
            total = start + end
            # In the crop branch the reference uses negative truncation of a positive cut.
            added = (int(delta * start / total) if total > 0 else
                     -(int(-delta) // 2) if crop_only else delta // 2)
            if axis == 'width':
                left += added; right += delta - added; target_width = aligned
            else:
                top += added; bottom += delta - added; target_height = aligned
    if min(target_width, target_height) < 1:
        raise ValueError('Outpaint alignment would produce an empty image; lower alignment or increase input dimensions.')
    source_x, source_y = max(0, -left), max(0, -top)
    dest_x, dest_y = max(0, left), max(0, top)
    copied_width = min(width - source_x, target_width - dest_x)
    copied_height = min(height - source_y, target_height - dest_y)
    source = image[:, source_y:source_y + copied_height, source_x:source_x + copied_width]
    if crop_only:
        output = source
        if output.shape[1:3] != (target_height, target_width):
            output = torch.nn.functional.interpolate(output.movedim(-1, 1), size=(target_height, target_width), mode='bilinear', align_corners=False).movedim(1, -1)
    else:
        output = torch.full((1, target_height, target_width, 3), 0.5, dtype=torch.float32)
        output[:, dest_y:dest_y + copied_height, dest_x:dest_x + copied_width] = source
    result_mask = torch.ones((1, target_height, target_width), dtype=torch.float32)
    if mask is None:
        if not crop_only:
            result_mask[:, dest_y:dest_y + copied_height, dest_x:dest_x + copied_width] = 0
    else:
        mask = mask.detach().cpu().float()
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError('Directional outpaint expects one MASK matching its single image.')
        if mask.shape[1:] != (height, width):
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False).squeeze(1)
        source_mask = mask[:, source_y:source_y + copied_height, source_x:source_x + copied_width]
        if crop_only:
            result_mask = source_mask
            if result_mask.shape[1:] != (target_height, target_width):
                result_mask = torch.nn.functional.interpolate(result_mask.unsqueeze(1), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(1)
        else:
            result_mask[:, dest_y:dest_y + copied_height, dest_x:dest_x + copied_width] = 1 - source_mask
    if feathering > 0:
        pil = Image.fromarray((result_mask[0].numpy() * 255).astype(np.uint8))
        result_mask = torch.from_numpy(np.array(pil.filter(ImageFilter.GaussianBlur(radius=feathering))).astype(np.float32) / 255).unsqueeze(0)
    return output, result_mask
