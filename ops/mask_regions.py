# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/遮罩混合运算.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import torch

class MaskCombineAlgorithm:

    def execute(self, 混合模式, BBOX, 对齐方式, 遮罩1=None, 遮罩2=None):
        if 遮罩1 is None and 遮罩2 is None:
            return (torch.zeros((1, 1024, 1024)), 0, 0)
        if 遮罩1 is None and 遮罩2 is not None:
            遮罩1 = torch.zeros_like(遮罩2)
        elif 遮罩2 is None and 遮罩1 is not None:
            遮罩2 = torch.zeros_like(遮罩1)
        elif 遮罩1.shape[1:] != 遮罩2.shape[1:]:
            raise ValueError('两个遮罩尺寸必须相同')
        h, w = (遮罩1.shape[1], 遮罩1.shape[2])
        result_mask = None
        if 混合模式 == '相加':
            result_mask = torch.clamp(遮罩1 + 遮罩2, 0, 1)
        elif 混合模式 == '相减':
            if torch.all(遮罩1 == 0):
                return (torch.zeros((1, h, w)), 0, 0)
            result_mask = torch.clamp(遮罩1 - torch.minimum(遮罩1, 遮罩2), 0, 1)
        elif 混合模式 == '相交':
            result_mask = torch.minimum(遮罩1, 遮罩2)
            if torch.all(result_mask == 0):
                return (torch.zeros((1, h, w)), 0, 0)
        elif 混合模式 == '排除':
            combined = torch.clamp(遮罩1 + 遮罩2, 0, 1)
            result_mask = 1 - combined
            if torch.all(result_mask == 0):
                return (torch.zeros((1, h, w)), 0, 0)
        elif 混合模式.startswith('水平'):

            def get_mask_range(mask):
                if torch.all(mask == 0):
                    return None
                non_zero = torch.nonzero(mask[0])
                if non_zero.numel() == 0:
                    return None
                return (non_zero[:, 1].min().item(), non_zero[:, 1].max().item())
            range1 = get_mask_range(遮罩1)
            range2 = get_mask_range(遮罩2)
            if range1 is None and range2 is None:
                return (torch.zeros((1, h, w)), 0, 0)
            elif range1 is None:
                result_mask = 遮罩2
            elif range2 is None:
                result_mask = 遮罩1
            else:
                x1_min, x1_max = range1
                x2_min, x2_max = range2
                cut_x = min(x1_max, x2_max)
                if 混合模式 == '水平取左':
                    result_mask = torch.zeros_like(遮罩1)
                    result_mask[:, :, :cut_x + 1] = 遮罩1[:, :, :cut_x + 1]
                else:
                    result_mask = torch.zeros_like(遮罩1)
                    result_mask[:, :, cut_x:] = 遮罩1[:, :, cut_x:]
        elif 混合模式.startswith('垂直'):

            def get_mask_range(mask):
                if torch.all(mask == 0):
                    return None
                non_zero = torch.nonzero(mask[0])
                if non_zero.numel() == 0:
                    return None
                return (non_zero[:, 0].min().item(), non_zero[:, 0].max().item())
            range1 = get_mask_range(遮罩1)
            range2 = get_mask_range(遮罩2)
            if range1 is None and range2 is None:
                return (torch.zeros((1, h, w)), 0, 0)
            elif range1 is None:
                result_mask = 遮罩2
            elif range2 is None:
                result_mask = 遮罩1
            else:
                y1_min, y1_max = range1
                y2_min, y2_max = range2
                cut_y = min(y1_max, y2_max)
                if 混合模式 == '垂直取上':
                    result_mask = torch.zeros_like(遮罩1)
                    result_mask[:, :cut_y + 1, :] = 遮罩1[:, :cut_y + 1, :]
                else:
                    result_mask = torch.zeros_like(遮罩1)
                    result_mask[:, cut_y:, :] = 遮罩1[:, cut_y:, :]
        if result_mask is None:
            result_mask = 遮罩1

        def get_mask_bbox(mask):
            if torch.all(mask == 0):
                return (0, 0, 0, 0)
            non_zero = torch.nonzero(mask[0])
            y_min, x_min = non_zero.min(dim=0)[0]
            y_max, x_max = non_zero.max(dim=0)[0]
            return (x_min.item(), y_min.item(), x_max.item(), y_max.item())
        x_min, y_min, x_max, y_max = get_mask_bbox(result_mask)
        rect_w = max(0, x_max - x_min + 1)
        rect_h = max(0, y_max - y_min + 1)
        if rect_w == 0 or rect_h == 0:
            return (result_mask, 0, 0)
        mask_width = rect_w
        mask_height = rect_h
        if BBOX != '关闭':
            if BBOX == '原始比例':
                new_w, new_h = (rect_w, rect_h)
            elif BBOX == '1：1长边不变':
                side = max(rect_w, rect_h)
                new_w, new_h = (side, side)
            elif BBOX == '1：1短边不变':
                side = min(rect_w, rect_h)
                new_w, new_h = (side, side)
            elif BBOX == '1：1宽度不变':
                new_w, new_h = (rect_w, rect_w)
            else:
                new_w, new_h = (rect_h, rect_h)
            mask_width = new_w
            mask_height = new_h
            if BBOX != '原始比例':
                if 对齐方式 == '左对齐':
                    new_x = x_min
                    new_y = y_min + (rect_h - new_h) // 2
                elif 对齐方式 == '右对齐':
                    new_x = x_min + (rect_w - new_w)
                    new_y = y_min + (rect_h - new_h) // 2
                elif 对齐方式 == '上对齐':
                    new_x = x_min + (rect_w - new_w) // 2
                    new_y = y_min
                elif 对齐方式 == '下对齐':
                    new_x = x_min + (rect_w - new_w) // 2
                    new_y = y_min + (rect_h - new_h)
                else:
                    new_x = x_min + (rect_w - new_w) // 2
                    new_y = y_min + (rect_h - new_h) // 2
            else:
                new_x = x_min
                new_y = y_min
            new_mask = torch.zeros((1, h, w))
            start_x = max(0, new_x)
            start_y = max(0, new_y)
            end_x = min(w, new_x + new_w)
            end_y = min(h, new_y + new_h)
            if start_x < end_x and start_y < end_y:
                new_mask[:, start_y:end_y, start_x:end_x] = 1
            result_mask = new_mask
        return (result_mask, mask_width, mask_height)
