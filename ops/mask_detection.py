# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/遮罩检测.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np

class MaskDetectAlgorithm:
    """
    孤海遮罩检测
    输入遮罩自动检测有效区域
    """

    def __init__(self):
        pass

    def detect(self, 遮罩, 过滤最小值):
        mask_np = 遮罩.cpu().numpy().squeeze()
        has_mask = np.any(mask_np > 0.5)
        if 过滤最小值 == 0:
            return (bool(has_mask),)
        if has_mask:
            valid_area = np.sum(mask_np > 0.5)
            total_pixels = mask_np.size
            area_ratio = valid_area / total_pixels * 100
            return (bool(area_ratio >= 过滤最小值),)
        return (False,)
