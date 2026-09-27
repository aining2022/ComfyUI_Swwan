# SPDX-License-Identifier: MIT
# Derived from ComfyUI_LayerStyle MaskGrow / expand_mask, chflame163; see MIT-LayerStyle.txt.
import numpy as np
import torch
from PIL import Image, ImageFilter
from .dependencies import lazy_module


def layer_grow(mask, invert_mask, grow, blur):
    ndimage = lazy_module("scipy.ndimage", "vision")
    results = []
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for m in mask:
        if invert_mask: m = 1 - m
        # The original interface quantizes before morphology and again via PIL after blur.
        out = np.clip(m.cpu().numpy() * 255., 0, 255).astype(np.uint8).astype(np.float32) / 255.
        for _ in range(abs(grow)):
            out = (ndimage.grey_erosion if grow < 0 else ndimage.grey_dilation)(out, footprint=footprint)
        pil = Image.fromarray(np.clip(out * 255., 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        results.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255.))
    return torch.stack(results)
