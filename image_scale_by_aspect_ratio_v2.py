# SPDX-License-Identifier: MIT
"""Legacy five-output and mask-only interface."""
from .ops.aspect_resize import aspect_resize

class ImageScaleByAspectRatioV2:

    def __init__(self):
        self.NODE_NAME = 'ImageScaleByAspectRatio V2'

    @classmethod
    def INPUT_TYPES(self):
        ratio_list = ['original', 'custom', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16']
        fit_mode = ['letterbox', 'crop', 'fill']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {
            "required": {
                "aspect_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "fit": (fit_mode,),
                "method": (method_mode,),
                "round_to_multiple": (multiple_list,),
                "scale_to_side": (scale_to_list,),  # 是否按长边缩放
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                "background_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "image": ("IMAGE",),  #
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT",)
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height",)
    FUNCTION = 'image_scale_by_aspect_ratio'
    CATEGORY = 'RunningHub/LayerUtility'

    def image_scale_by_aspect_ratio(self, aspect_ratio, proportional_width, proportional_height,
                                    fit, method, round_to_multiple, scale_to_side, scale_to_length,
                                    background_color, image=None, mask=None):
        return aspect_resize(aspect_ratio, proportional_width, proportional_height, fit, method,
                             round_to_multiple, scale_to_side, scale_to_length, background_color, image, mask)
