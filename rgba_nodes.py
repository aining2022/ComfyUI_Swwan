import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


def _build_png_metadata(prompt, extra_pnginfo):
    if args.disable_metadata:
        return None

    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(value))
    return metadata


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


def _unpremultiply(image, alpha, epsilon):
    safe_alpha = alpha.clamp(min=epsilon)
    image_out = image / safe_alpha.unsqueeze(-1)
    image_out = torch.where(alpha.unsqueeze(-1) > epsilon, image_out, torch.zeros_like(image_out))
    return image_out.clamp(0.0, 1.0)


def _flatten_premultiplied(image, alpha, background):
    return (image + (1.0 - alpha).unsqueeze(-1) * background.view(1, 1, 1, 3)).clamp(0.0, 1.0)


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
        return (_unpremultiply(image, alpha_out, epsilon), alpha_out)


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

            metadata = _build_png_metadata(prompt, extra_pnginfo)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter + batch_number:05}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {"ui": {"images": results}}


class RGBA_Multi_Save:
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
                "has_alpha": ("BOOLEAN",),
                "file_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "alpha_mode": (["auto", "keep", "flatten"], {"default": "auto"}),
                "filename_prefix": ("STRING", {"default": "RGBA"}),
            },
            "optional": {
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.000001, "max": 0.1, "step": 0.0005}),
                "background_red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_green": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_blue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "webp_quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "webp_lossless": ("BOOLEAN", {"default": False}),
                "png_compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Swwan/RGBA"
    DESCRIPTION = """
Saves premultiplied IMAGE output as JPEG, PNG, or WebP.
Keeps alpha only when the selected format supports it and alpha_mode requires it.
"""

    def save_images(
        self,
        image,
        alpha,
        has_alpha,
        file_format,
        alpha_mode,
        filename_prefix="RGBA",
        epsilon=0.001,
        background_red=1.0,
        background_green=1.0,
        background_blue=1.0,
        jpeg_quality=95,
        webp_quality=90,
        webp_lossless=False,
        png_compress_level=4,
        prompt=None,
        extra_pnginfo=None,
    ):
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
        background = torch.tensor(
            [background_red, background_green, background_blue],
            device=image.device,
            dtype=torch.float32,
        )

        supports_alpha = file_format in {"png", "webp"}
        if alpha_mode == "keep":
            keep_alpha = supports_alpha and bool(has_alpha)
        elif alpha_mode == "flatten":
            keep_alpha = False
        else:
            keep_alpha = supports_alpha and bool(has_alpha)

        if keep_alpha:
            output_image = _unpremultiply(image, alpha, epsilon)
        else:
            output_image = _flatten_premultiplied(image, alpha, background)

        metadata = _build_png_metadata(prompt, extra_pnginfo) if file_format == "png" else None
        extension = "jpg" if file_format == "jpeg" else file_format
        results = []

        for batch_number, (rgb_tensor, alpha_tensor) in enumerate(zip(output_image, alpha)):
            rgb = np.clip(rgb_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

            if keep_alpha:
                alpha_np = np.clip(alpha_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(np.dstack((rgb, alpha_np)), mode="RGBA")
            else:
                pil_image = Image.fromarray(rgb, mode="RGB")

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter + batch_number:05}.{extension}"
            output_path = os.path.join(full_output_folder, file)

            if file_format == "jpeg":
                pil_image = pil_image.convert("RGB")
                pil_image.save(output_path, format="JPEG", quality=jpeg_quality)
            elif file_format == "webp":
                pil_image.save(
                    output_path,
                    format="WEBP",
                    quality=webp_quality,
                    lossless=webp_lossless,
                )
            else:
                pil_image.save(output_path, format="PNG", pnginfo=metadata, compress_level=png_compress_level)

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
    "RGBA_Multi_Save": RGBA_Multi_Save,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBA_Safe_Pre": "RGBA Safe Pre (Swwan)",
    "RGBA_Safe_Post": "RGBA Safe Post (Swwan)",
    "RGBA_Save": "RGBA Save (Swwan)",
    "RGBA_Multi_Save": "RGBA Multi Save (Swwan)",
}
