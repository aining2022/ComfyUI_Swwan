import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


def _normalize_mask(mask, batch_size, device):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]

    mask = mask.to(device=device, dtype=torch.float32)
    if mask.shape[0] != batch_size:
        indices = torch.arange(batch_size, device=device) % mask.shape[0]
        mask = mask.index_select(0, indices)
    return mask.clamp(0.0, 1.0)


def _has_meaningful_alpha(mask, image_height, image_width):
    mask_height, mask_width = mask.shape[-2], mask.shape[-1]
    if mask_height != image_height or mask_width != image_width:
        return False

    if torch.any(mask > 0):
        return True

    # ComfyUI uses a 64x64 all-zero fallback mask when the source image has no alpha.
    return (mask_width, mask_height) != (64, 64)


def _resize_alpha(alpha, batch_size, height, width):
    alpha = _normalize_mask(alpha, batch_size, alpha.device)
    if alpha.shape[-2:] == (height, width):
        return alpha.clamp(0.0, 1.0)

    resized = F.interpolate(
        alpha.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
    return resized.clamp(0.0, 1.0)


class RGBA_Safe_Pre:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image_out", "alpha", "has_alpha")
    FUNCTION = "prepare"
    CATEGORY = "Swwan/RGBA"
    DESCRIPTION = """
Extracts alpha from ComfyUI's Load Image mask, converts it back to alpha,
and premultiplies RGB so the image can safely pass through RGB-only nodes.
"""

    def prepare(self, image, mask):
        image = image.to(dtype=torch.float32)
        batch_size, image_height, image_width, _ = image.shape
        mask = _normalize_mask(mask, batch_size, image.device)

        has_alpha = _has_meaningful_alpha(mask, image_height, image_width)
        if has_alpha:
            alpha = (1.0 - mask).clamp(0.0, 1.0)
            image_out = image * alpha.unsqueeze(-1)
        else:
            alpha = torch.ones((batch_size, image_height, image_width), device=image.device, dtype=torch.float32)
            image_out = image

        return (image_out.clamp(0.0, 1.0), alpha, bool(has_alpha))


class RGBA_Safe_Post:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "has_alpha": ("BOOLEAN",),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.1, "step": 0.0005}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_out", "alpha_out")
    FUNCTION = "restore"
    CATEGORY = "Swwan/RGBA"
    DESCRIPTION = """
Resizes alpha to the processed image size, safely unpremultiplies RGB,
and returns an alpha mask ready for RGBA export.
"""

    def restore(self, image, alpha, has_alpha, epsilon=0.001):
        image = image.to(dtype=torch.float32)
        batch_size, image_height, image_width, _ = image.shape

        if not has_alpha:
            alpha_out = torch.ones((batch_size, image_height, image_width), device=image.device, dtype=torch.float32)
            return (image.clamp(0.0, 1.0), alpha_out)

        alpha = alpha.to(device=image.device, dtype=torch.float32)
        alpha_out = _resize_alpha(alpha, batch_size, image_height, image_width)
        safe_alpha = alpha_out.clamp(min=epsilon)
        image_out = image / safe_alpha.unsqueeze(-1)
        image_out = torch.where(alpha_out.unsqueeze(-1) > epsilon, image_out, torch.zeros_like(image_out))
        return (image_out.clamp(0.0, 1.0), alpha_out)


class RGBA_Save:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "filename_prefix": ("STRING", {"default": "RGBA"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_rgba"
    OUTPUT_NODE = True
    CATEGORY = "Swwan/RGBA"
    DESCRIPTION = """
Saves RGB plus alpha as an RGBA PNG without dropping transparency.
"""

    def save_rgba(self, image, alpha, filename_prefix="RGBA", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            image[0].shape[1],
            image[0].shape[0],
        )

        image = image.to(dtype=torch.float32)
        alpha = alpha.to(device=image.device, dtype=torch.float32)
        alpha = _resize_alpha(alpha, image.shape[0], image.shape[1], image.shape[2])

        results = []
        for batch_number, (rgb_tensor, alpha_tensor) in enumerate(zip(image, alpha)):
            rgb = np.clip(rgb_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            alpha_np = np.clip(alpha_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

            rgba = np.dstack((rgb, alpha_np))
            img = Image.fromarray(rgba, mode="RGBA")

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter + batch_number:05}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "RGBA_Safe_Pre": RGBA_Safe_Pre,
    "RGBA_Safe_Post": RGBA_Safe_Post,
    "RGBA_Save": RGBA_Save,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBA_Safe_Pre": "RGBA Safe Pre (Swwan)",
    "RGBA_Safe_Post": "RGBA Safe Post (Swwan)",
    "RGBA_Save": "RGBA Save (Swwan)",
}
