# SPDX-License-Identifier: MIT
"""Describe aligned preprocessing using the existing GH_CropRestore SEAM contract.

This is metadata conversion only: GH performs inverse resize, color restoration and
feathering. No BlueBox/Swwan restoration algorithm is called.
"""
import torch


def to_goohai_seam(seam):
    if seam.get('qwen21_geometry_version') != 1:
        raise ValueError('Expected qwen21_aligned preprocessing metadata')
    w, h = seam['original_size']
    x0, y0, x1, y1 = seam['virtual_box']
    tw, th = seam['canvas_size']
    px, py, pr, pb = seam['content_box']
    mask = seam['original_mask']
    ys, xs = torch.where(mask[0] > 0)
    return {
        '原始图像': seam['original_image'], '原始遮罩': mask, '填充后遮罩': mask,
        '原像素': (h, w), '裁剪区域': (y0, y1, x0, x1), '目标尺寸': (th, tw),
        '缩放比例宽': 1.0, '缩放比例高': 1.0,
        '最终缩放宽': (pr-px)/(x1-x0), '最终缩放高': (pb-py)/(y1-y0),
        '输出尺寸模式': '自定义宽高',
        '填充信息': (px, py, pr-px, pb-py, tw, th),
        '遮罩填充': False, '遮罩白色区域宽': int(xs.max()-xs.min())+1,
        '遮罩白色区域高': int(ys.max()-ys.min())+1,
        # Diagnostics only; GH consumes the fields above, not this native record.
        'swwan_preprocess': seam,
    }
