"""
Image Comparer Node - 移植自 rgthree-comfy
在 UI 中比较两张图片，支持滑动或点击切换
"""

import folder_paths
import os
from PIL import Image
import numpy as np
import torch


class RgthreeImageComparer:
    """图片比较节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    CATEGORY = "rgthree"
    OUTPUT_NODE = True

    def compare_images(
        self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None
    ):
        """比较并保存图片"""
        results = []

        # 保存图片的辅助函数
        def save_images(images, prefix):
            saved = []
            if images is None or len(images) == 0:
                return saved

            output_dir = folder_paths.get_temp_directory()

            for batch_number, image in enumerate(images):
                # 转换 tensor 到 PIL Image
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # 生成文件名
                filename = f"{prefix}_{batch_number:05d}.png"
                filepath = os.path.join(output_dir, filename)

                # 保存图片
                img.save(filepath, compress_level=4)

                saved.append(
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                    }
                )

            return saved

        result = {"ui": {"a_images": [], "b_images": []}}

        if image_a is not None and len(image_a) > 0:
            result["ui"]["a_images"] = save_images(image_a, "rgthree_compare_a")

        if image_b is not None and len(image_b) > 0:
            result["ui"]["b_images"] = save_images(image_b, "rgthree_compare_b")

        return result


NODE_CLASS_MAPPINGS = {"Image Comparer (rgthree)": RgthreeImageComparer}

NODE_DISPLAY_NAME_MAPPINGS = {"Image Comparer (rgthree)": "Image Comparer"}
