# SPDX-License-Identifier: GPL-3.0-only
"""Pure mask/region tasks: shared operations instead of per-plugin menu duplicates."""
import numpy as np
import torch
from ..ops.dependencies import lazy_module


class MaskProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "operation": (["cleanup", "fill_holes", "binary", "layer_grow", "grow_blur"],)},
                "optional": {
                    "threshold": ("INT", {"default": 20, "min": 1, "max": 255}),
                    "erode_dilate": ("INT", {"default": 0, "min": -256, "max": 256}),
                    "close_holes": ("INT", {"default": 0, "min": 0, "max": 128}),
                    "remove_isolated_pixels": ("INT", {"default": 0, "min": 0, "max": 32}),
                    "smooth": ("INT", {"default": 0, "min": 0, "max": 256}),
                    "blur": ("INT", {"default": 0, "min": 0, "max": 999}),
                    "invert_mask": ("BOOLEAN", {"default": False}),
                    "grow": ("INT", {"default": 4, "min": -999, "max": 999}),
                    "expand": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                    "incremental_expandrate": ("FLOAT", {"default": 0., "min": 0., "max": 100.}),
                    "tapered_corners": ("BOOLEAN", {"default": True}),
                    "flip_input": ("BOOLEAN", {"default": False}),
                    "blur_radius": ("FLOAT", {"default": 0., "min": 0., "max": 100.}),
                    "lerp_alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1.}),
                    "decay_factor": ("FLOAT", {"default": 1., "min": 0., "max": 1.}),
                    "fill_holes": ("BOOLEAN", {"default": False}),
                }}
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "process"
    DESCRIPTION = "Cleanup, hard threshold, hole filling and two distinct growth algorithms. Fill holes retains the historical [B,1,H,W] output; other modes return [B,H,W]."

    def process(self, mask, operation="cleanup", threshold=20, erode_dilate=0, close_holes=0,
                remove_isolated_pixels=0, smooth=0, blur=0, invert_mask=False, grow=4,
                expand=0, incremental_expandrate=0., tapered_corners=True, flip_input=False,
                blur_radius=0., lerp_alpha=1., decay_factor=1., fill_holes=False):
        if mask.ndim == 4: mask = mask.squeeze(1)
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        if mask.ndim != 3 or len(mask) == 0:
            raise ValueError("Mask Process expects a non-empty [B,H,W] mask batch")
        if operation == "binary":
            out = mask.clone().cpu()
            out[out > threshold / 255.] = 1.
            out[out <= threshold / 255.] = 0.
        elif operation == "fill_holes":
            ndimage = lazy_module("scipy.ndimage", "vision")
            # WAS quantizes before filling and returns a singleton channel dimension.
            out = torch.stack([torch.from_numpy(ndimage.binary_fill_holes(
                np.clip(m.cpu().numpy()*255, 0, 255).astype(np.uint8) > 0).astype(np.float32)) for m in mask]).unsqueeze(1)
        elif operation == "layer_grow":
            from ..ops.mask_processing import layer_grow
            out = layer_grow(mask, invert_mask, grow, blur)
        elif operation == "grow_blur":
            from ..ops.mask_growth import GrowMaskAlgorithm
            try:
                return GrowMaskAlgorithm().expand_mask(mask, expand, tapered_corners, flip_input, blur_radius,
                                                      incremental_expandrate, lerp_alpha, decay_factor, fill_holes)
            except ImportError as exc:
                raise RuntimeError("Grow Blur requires kornia and scipy (ComfyUI / vision dependencies)") from exc
        elif operation == "cleanup":
            from ..ops.mask_cleanup import MaskCleanupAlgorithm
            try:
                out = MaskCleanupAlgorithm().execute(mask, erode_dilate, smooth, remove_isolated_pixels, blur, close_holes)[0]
            except ImportError as exc:
                raise RuntimeError("Mask Cleanup requires scipy and torchvision (vision dependencies)") from exc
        else: raise ValueError(f"Unknown mask operation: {operation}")
        return out, 1. - out


class MaskCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "operation": (["相加", "相减", "相交", "排除", "水平取左", "水平取右", "垂直取上", "垂直取下"],),
            "bbox_mode": (["关闭", "原始比例", "1：1长边不变", "1：1短边不变", "1：1宽度不变", "1：1高度不变"],),
            "alignment": (["左对齐", "右对齐", "居中", "上对齐", "下对齐"],)},
            "optional": {"mask_1": ("MASK",), "mask_2": ("MASK",)}}
    RETURN_TYPES = ("MASK", "INT", "INT")
    RETURN_NAMES = ("mask", "width", "height")
    FUNCTION = "combine"
    DESCRIPTION = "Mask arithmetic and aligned BBOX crops. Width/height are the effective region, not necessarily canvas dimensions."

    def combine(self, operation, bbox_mode, alignment, mask_1=None, mask_2=None):
        from ..ops.mask_regions import MaskCombineAlgorithm
        return MaskCombineAlgorithm().execute(operation, bbox_mode, alignment, mask_1, mask_2)


class MaskAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}, "optional": {
            **{name: ("INT", {"default": 0, "min": -90, "max": 1000}) for name in
               ("top_percent", "bottom_percent", "left_percent", "right_percent")},
            "minimum_area_percent": ("FLOAT", {"default": 3., "min": 0., "max": 10., "step": .1})}}
    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("mask", "canvas_width", "canvas_height", "mask_width", "mask_height", "center_x", "center_y", "has_mask")
    FUNCTION = "analyze"
    DESCRIPTION = "Bounds use the first mask and >=0.5; area detection uses all masks and >0.5. Empty bounds retain the reference full-canvas fallback."

    def analyze(self, mask, top_percent=0, bottom_percent=0, left_percent=0, right_percent=0, minimum_area_percent=3.):
        from ..ops.mask_analysis import MaskAnalysisAlgorithm
        from ..ops.mask_detection import MaskDetectAlgorithm
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        if mask.ndim != 3 or len(mask) == 0: raise ValueError("Mask Analyze expects a non-empty [B,H,W] mask batch")
        return MaskAnalysisAlgorithm().analyze_mask(mask, top_percent, bottom_percent, left_percent, right_percent) + MaskDetectAlgorithm().detect(mask, minimum_area_percent)


class MaskSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"operation": (["from_mask", "combined_mask", "mask_batch", "filter"],)},
                "optional": {"mask": ("MASK",), "segs": ("SEGS",),
                    "combined": ("BOOLEAN", {"default": False}),
                    "crop_factor": ("FLOAT", {"default": 3., "min": 1., "max": 100.}),
                    "bbox_fill": ("BOOLEAN", {"default": False}),
                    "drop_size": ("INT", {"default": 10, "min": 1, "max": 16384}),
                    "contour_fill": ("BOOLEAN", {"default": False}),
                    "sort_rule": (["面积大小", "宽度大小", "高度大小", "左右上下", "左右下上"],),
                    "sort_order": (["正序", "反序"],),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9223372036854775807}),
                    "count": ("INT", {"default": 1, "min": 0, "max": 9223372036854775807}),
                    "group_threshold": ("INT", {"default": 50, "min": 1, "max": 500})}}
    RETURN_TYPES = ("SEGS", "MASK")
    RETURN_NAMES = ("segs", "mask")
    FUNCTION = "process"
    DESCRIPTION = "Pure Impact-compatible seven-field SEGS processing. Combined and batch masks keep byte quantization; filtering keeps soft masks, first-canvas coordinates and cyclic positive-count indexing."

    def process(self, operation, mask=None, segs=None, combined=False, crop_factor=3., bbox_fill=False,
                drop_size=10, contour_fill=False, sort_rule="面积大小", sort_order="正序", start_index=0, count=1, group_threshold=50):
        from ..ops import segments
        if operation == "from_mask":
            if mask is None: raise ValueError("Mask Segments: from_mask requires mask")
            if mask.ndim == 4: mask = mask.squeeze(0).squeeze(0)
            elif mask.ndim == 3: mask = mask.squeeze(0)
            segs = segments.mask_to_segs(mask.cpu(), combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
        elif segs is None: raise ValueError(f"Mask Segments: {operation} requires SEGS")
        if operation == "filter":
            from ..ops.segment_order import SegmentOrderAlgorithm
            return SegmentOrderAlgorithm().filter_segments(segs, sort_rule, sort_order, start_index, count, group_threshold)
        if operation == "mask_batch":
            return segs, torch.stack(segments.segs_to_masklist(segs))
        if operation not in ("from_mask", "combined_mask"): raise ValueError(f"Unknown SEGS operation: {operation}")
        return segs, segments.segs_to_combined_mask(segs).unsqueeze(0)


class ImageMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}, "optional": {
            "fill_holes": ("BOOLEAN", {"default": True}), "crop_mask": ("BOOLEAN", {"default": True}),
            "crop_factor": ("FLOAT", {"default": 1.2, "min": 1., "max": 2., "step": .1}),
            "stroke_width": ("INT", {"default": 0, "min": 0, "max": 512}),
            "background_color": ("COLORCODE", {"default": "#364254"})}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "rgb", "mask")
    FUNCTION = "matte"
    DESCRIPTION = "Use an existing mask as alpha; optionally fill holes, crop and add a stroke. RGB fills transparent pixels with background_color. This does not run background detection or premultiply RGB."

    def matte(self, image, mask, fill_holes=True, crop_mask=True, crop_factor=1.2, stroke_width=0, background_color="#364254"):
        from ..ops.image_matte import MaskMatteAlgorithm
        return MaskMatteAlgorithm().remove_background(image, mask, fill_holes, crop_mask, crop_factor, stroke_width, background_color)
