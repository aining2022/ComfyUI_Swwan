# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/遮罩分析.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np
import torch

class MaskAnalysisAlgorithm:

    def analyze_mask(self, 遮罩, 上扩百分比, 下扩百分比, 左扩百分比, 右扩百分比):
        mask_np = 遮罩.cpu().numpy()[0]
        orig_h, orig_w = mask_np.shape
        new_mask = np.zeros_like(mask_np)
        y_indices, x_indices = np.where(mask_np >= 0.5)
        if len(x_indices) == 0 or len(y_indices) == 0:
            new_mask = mask_np
            return (遮罩, int(orig_w), int(orig_h), int(orig_w), int(orig_h), int(orig_w // 2), int(orig_h // 2))
        x_min, x_max = (np.min(x_indices), np.max(x_indices))
        y_min, y_max = (np.min(y_indices), np.max(y_indices))
        orig_mask_w = x_max - x_min + 1
        orig_mask_h = y_max - y_min + 1
        top_ext = 0
        bottom_ext = 0
        left_ext = 0
        right_ext = 0
        if 上扩百分比 < 0 or 下扩百分比 < 0:
            total_vertical_shrink = max(0, -上扩百分比) + max(0, -下扩百分比)
            if total_vertical_shrink > 0:
                max_shrink_h = orig_mask_h * 0.9
                if total_vertical_shrink > max_shrink_h:
                    scale = max_shrink_h / total_vertical_shrink
                    top_ext = -int(max(0, -上扩百分比) * scale)
                    bottom_ext = -int(max(0, -下扩百分比) * scale)
                else:
                    top_ext = 上扩百分比
                    bottom_ext = 下扩百分比
        else:
            top_ext = 上扩百分比
            bottom_ext = 下扩百分比
        if 左扩百分比 < 0 or 右扩百分比 < 0:
            total_horizontal_shrink = max(0, -左扩百分比) + max(0, -右扩百分比)
            if total_horizontal_shrink > 0:
                max_shrink_w = orig_mask_w * 0.9
                if total_horizontal_shrink > max_shrink_w:
                    scale = max_shrink_w / total_horizontal_shrink
                    left_ext = -int(max(0, -左扩百分比) * scale)
                    right_ext = -int(max(0, -右扩百分比) * scale)
                else:
                    left_ext = 左扩百分比
                    right_ext = 右扩百分比
        else:
            left_ext = 左扩百分比
            right_ext = 右扩百分比
        top_ext_val = int(orig_mask_h * abs(top_ext) / 100) * (-1 if top_ext < 0 else 1)
        bottom_ext_val = int(orig_mask_h * abs(bottom_ext) / 100) * (-1 if bottom_ext < 0 else 1)
        left_ext_val = int(orig_mask_w * abs(left_ext) / 100) * (-1 if left_ext < 0 else 1)
        right_ext_val = int(orig_mask_w * abs(right_ext) / 100) * (-1 if right_ext < 0 else 1)
        new_y_min = max(0, y_min - (top_ext_val if top_ext_val > 0 else 0) + (abs(top_ext_val) if top_ext_val < 0 else 0))
        new_y_max = min(orig_h - 1, y_max + (bottom_ext_val if bottom_ext_val > 0 else 0) - (abs(bottom_ext_val) if bottom_ext_val < 0 else 0))
        new_x_min = max(0, x_min - (left_ext_val if left_ext_val > 0 else 0) + (abs(left_ext_val) if left_ext_val < 0 else 0))
        new_x_max = min(orig_w - 1, x_max + (right_ext_val if right_ext_val > 0 else 0) - (abs(right_ext_val) if right_ext_val < 0 else 0))
        if new_y_min > new_y_max:
            new_y_min, new_y_max = (y_min, y_max)
        if new_x_min > new_x_max:
            new_x_min, new_x_max = (x_min, x_max)
        new_mask[new_y_min:new_y_max + 1, new_x_min:new_x_max + 1] = 1.0
        new_mask_tensor = torch.from_numpy(new_mask).unsqueeze(0)
        new_mask_w = new_x_max - new_x_min + 1
        new_mask_h = new_y_max - new_y_min + 1
        new_center_x = (new_x_min + new_x_max) // 2
        new_center_y = (new_y_min + new_y_max) // 2
        return (new_mask_tensor, int(orig_w), int(orig_h), int(new_mask_w), int(new_mask_h), int(new_center_x), int(new_center_y))
