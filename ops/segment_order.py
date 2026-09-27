# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/Seg次序过滤.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np
import torch
from typing import List, Tuple

class SegmentOrderAlgorithm:

    def get_segment_area(self, seg) -> float:
        """计算分割区域的面积"""
        if hasattr(seg, 'cropped_mask') and seg.cropped_mask is not None:
            mask = seg.cropped_mask
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 3:
                    mask = mask[0]
                return float(torch.sum(mask > 0.1).item())
            elif isinstance(mask, np.ndarray):
                return float(np.sum(mask > 0.1))
        return 0.0

    def get_segment_bbox(self, seg) -> tuple:
        """获取分割区域的边界框信息"""
        if hasattr(seg, 'bbox'):
            bbox = seg.bbox
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return bbox
        return (0, 0, 0, 0)

    def get_segment_mask(self, seg) -> torch.Tensor:
        """获取分割区域的掩码"""
        if hasattr(seg, 'cropped_mask') and seg.cropped_mask is not None:
            mask = seg.cropped_mask
            if isinstance(mask, torch.Tensor):
                return mask
            elif isinstance(mask, np.ndarray):
                return torch.from_numpy(mask)
        return torch.zeros((1, 512, 512), dtype=torch.float32)

    def get_segment_feature(self, seg, feature_type: str) -> float:
        """根据特征类型获取分割区域的特征值"""
        bbox = self.get_segment_bbox(seg)
        if not bbox or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if feature_type == '面积大小':
            return self.get_segment_area(seg)
        elif feature_type == '宽度大小':
            return float(width)
        elif feature_type == '高度大小':
            return float(height)
        elif feature_type == '左右上下':
            return float(x1) * 10000 + float(y1)
        elif feature_type == '左右下上':
            return float(x1) * 10000 + float(y2)
        return 0.0

    def group_segments_by_x(self, segs_list: List, threshold: int) -> List[List]:
        """根据X1坐标将分割区域分组"""
        if not segs_list:
            return []
        sorted_by_x = sorted(segs_list, key=lambda seg: self.get_segment_bbox(seg)[0])
        groups = []
        current_group = []
        current_x = None
        for seg in sorted_by_x:
            x1, y1, x2, y2 = self.get_segment_bbox(seg)
            if current_x is None:
                current_group.append(seg)
                current_x = x1
            elif abs(x1 - current_x) <= threshold:
                current_group.append(seg)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [seg]
                current_x = x1
        if current_group:
            groups.append(current_group)
        return groups

    def sort_groups_left_right_top_bottom(self, groups: List[List], reverse_order: bool) -> List:
        """左右上下排序：组从左到右，组内从上到下"""
        groups_sorted = sorted(groups, key=lambda group: np.mean([self.get_segment_bbox(seg)[0] for seg in group]))
        result = []
        for group in groups_sorted:
            group_sorted = sorted(group, key=lambda seg: self.get_segment_bbox(seg)[1])
            result.extend(group_sorted)
        if reverse_order:
            result = list(reversed(result))
        return result

    def sort_groups_left_right_bottom_top(self, groups: List[List], reverse_order: bool) -> List:
        """左右下上排序：组从左到右，组内从下到上"""
        groups_sorted = sorted(groups, key=lambda group: np.mean([self.get_segment_bbox(seg)[0] for seg in group]))
        result = []
        for group in groups_sorted:
            group_sorted = sorted(group, key=lambda seg: self.get_segment_bbox(seg)[3], reverse=True)
            result.extend(group_sorted)
        if reverse_order:
            result = list(reversed(result))
        return result

    def create_full_mask(self, shape, segs, crop_regions) -> torch.Tensor:
        """创建完整的全图mask，将各个seg的mask放置到正确位置"""
        h, w = shape
        full_mask = torch.zeros((h, w), dtype=torch.float32)
        for i, seg in enumerate(segs):
            crop_region = crop_regions[i]
            mask = self.get_segment_mask(seg)
            if mask.dim() == 3:
                mask = mask[0] if mask.shape[0] == 1 else mask
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            x1, y1, x2, y2 = crop_region
            crop_h, crop_w = (y2 - y1, x2 - x1)
            if mask.shape != (crop_h, crop_w):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(crop_h, crop_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if y1 >= 0 and y2 <= h and (x1 >= 0) and (x2 <= w):
                full_mask[y1:y2, x1:x2] = torch.maximum(full_mask[y1:y2, x1:x2], mask)
        return full_mask

    def filter_segments(self, Seg, 优先规则: str, 正反顺序: str, 开始索引: int, 过滤数量: int, 分组阈值: int) -> Tuple[tuple, torch.Tensor]:
        if isinstance(Seg, tuple) and len(Seg) == 2:
            image_shape, segs_list = Seg
        else:
            image_shape = (0, 0)
            segs_list = []
            if isinstance(Seg, list):
                segs_list = Seg
        if not segs_list:
            h, w = image_shape if image_shape != (0, 0) else (512, 512)
            empty_mask = torch.zeros((h, w), dtype=torch.float32)
            empty_segs = (image_shape, [])
            return (empty_segs, empty_mask)
        reverse_sort = 正反顺序 == '正序'
        if 优先规则 == '面积大小':
            sorted_segs = sorted(segs_list, key=lambda seg: self.get_segment_feature(seg, '面积大小'), reverse=reverse_sort)
        elif 优先规则 == '宽度大小':
            sorted_segs = sorted(segs_list, key=lambda seg: self.get_segment_feature(seg, '宽度大小'), reverse=reverse_sort)
        elif 优先规则 == '高度大小':
            sorted_segs = sorted(segs_list, key=lambda seg: self.get_segment_feature(seg, '高度大小'), reverse=reverse_sort)
        elif 优先规则 == '左右上下':
            groups = self.group_segments_by_x(segs_list, 分组阈值)
            sorted_segs = self.sort_groups_left_right_top_bottom(groups, not reverse_sort)
        elif 优先规则 == '左右下上':
            groups = self.group_segments_by_x(segs_list, 分组阈值)
            sorted_segs = self.sort_groups_left_right_bottom_top(groups, not reverse_sort)
        else:
            sorted_segs = segs_list
        num_segments = len(sorted_segs)
        if 过滤数量 == 0:
            if 开始索引 >= num_segments:
                h, w = image_shape
                empty_mask = torch.zeros((h, w), dtype=torch.float32)
                empty_segs = (image_shape, [])
                return (empty_segs, empty_mask)
            selected_indices = list(range(开始索引, num_segments))
            selected_segs = [sorted_segs[idx] for idx in selected_indices]
            new_segs = (image_shape, selected_segs)
            if not selected_segs:
                h, w = image_shape
                empty_mask = torch.zeros((h, w), dtype=torch.float32)
                return (new_segs, empty_mask)
            elif len(selected_segs) == 1:
                seg = selected_segs[0]
                mask = self.get_segment_mask(seg)
                crop_region = seg.crop_region
                if mask.dim() == 3:
                    mask = mask[0] if mask.shape[0] == 1 else mask
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                x1, y1, x2, y2 = crop_region
                crop_h, crop_w = (y2 - y1, x2 - x1)
                if mask.shape != (crop_h, crop_w):
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(crop_h, crop_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                h, w = image_shape
                full_mask = torch.zeros((h, w), dtype=torch.float32)
                if y1 >= 0 and y2 <= h and (x1 >= 0) and (x2 <= w):
                    full_mask[y1:y2, x1:x2] = mask
                merged_mask = full_mask
            else:
                crop_regions = [seg.crop_region for seg in selected_segs]
                merged_mask = self.create_full_mask(image_shape, selected_segs, crop_regions)
            return (new_segs, merged_mask)
        过滤数量 = min(过滤数量, num_segments)
        selected_indices = []
        for i in range(过滤数量):
            idx = (开始索引 + i) % num_segments
            selected_indices.append(idx)
        selected_segs = [sorted_segs[idx] for idx in selected_indices]
        new_segs = (image_shape, selected_segs)
        h, w = image_shape
        if 过滤数量 == 1:
            seg = selected_segs[0]
            mask = self.get_segment_mask(seg)
            crop_region = seg.crop_region
            if mask.dim() == 3:
                mask = mask[0] if mask.shape[0] == 1 else mask
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            x1, y1, x2, y2 = crop_region
            crop_h, crop_w = (y2 - y1, x2 - x1)
            if mask.shape != (crop_h, crop_w):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(crop_h, crop_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            full_mask = torch.zeros((h, w), dtype=torch.float32)
            if y1 >= 0 and y2 <= h and (x1 >= 0) and (x2 <= w):
                full_mask[y1:y2, x1:x2] = mask
            merged_mask = full_mask
        else:
            crop_regions = [seg.crop_region for seg in selected_segs]
            merged_mask = self.create_full_mask(image_shape, selected_segs, crop_regions)
        return (new_segs, merged_mask)
