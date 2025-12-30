"""
Image Comparer Node - Integrated into ComfyUI_Swwan
Compare two images in the UI with a slider or click.
"""

import folder_paths
import os
import numpy as np
import torch
import random
from PIL import Image

class RgthreeImageComparer:
    """A node that compares two images in the UI with a custom widget."""

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
    CATEGORY = "Swwan/image"
    OUTPUT_NODE = True

    def compare_images(self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None):
        results = {"ui": {"a_images": [], "b_images": []}}
        
        # Use a unique prefix to avoid browser caching issues
        unique_prefix = f"comparer_{random.randint(0, 1000000)}"

        def save_to_temp(images, sub_prefix):
            saved = []
            if images is None or len(images) == 0:
                return saved
            
            # Ensure images is a CPU tensor
            images = images.cpu()
            temp_dir = folder_paths.get_temp_directory()
            
            for i, img_tensor in enumerate(images):
                # Convert [0, 1] tensor to [0, 255] PIL image
                img_np = 255.0 * img_tensor.numpy()
                img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                
                filename = f"{unique_prefix}_{sub_prefix}_{i:05d}.png"
                filepath = os.path.join(temp_dir, filename)
                
                img_pil.save(filepath, compress_level=4)
                
                saved.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp",
                })
            return saved

        if image_a is not None:
            results["ui"]["a_images"] = save_to_temp(image_a, "a")
        
        if image_b is not None:
            results["ui"]["b_images"] = save_to_temp(image_b, "b")

        return results

NODE_CLASS_MAPPINGS = {"Image Comparer (rgthree)": RgthreeImageComparer}
NODE_DISPLAY_NAME_MAPPINGS = {"Image Comparer (rgthree)": "Image Comparer (Swwan)"}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
