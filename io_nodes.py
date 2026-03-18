import os
import json
import re
import numpy as np
from PIL import Image
import folder_paths


FORMAT_EXTENSION_MAP = {
    "png": "png",
    "webp": "webp",
    "jpg": "jpg",
    "tif": "tif",
    "bmp": "bmp",
}

PIL_FORMAT_MAP = {
    "png": "PNG",
    "webp": "WEBP",
    "jpg": "JPEG",
    "tif": "TIFF",
    "bmp": "BMP",
}

FAST_DEFAULTS = {
    "quality": 88,
    "png_compress_level": 1,
    "webp_method": 2,
}


def _resolve_output_paths(output_path, batch_size):
    output_dir = folder_paths.get_output_directory()

    if isinstance(output_path, str):
        resolved_output_path = output_path or output_dir
        os.makedirs(resolved_output_path, exist_ok=True)
        return [resolved_output_path] * batch_size

    if isinstance(output_path, list) and len(output_path) == batch_size:
        for path in output_path:
            os.makedirs(path, exist_ok=True)
        return output_path

    print("Invalid output_path format. Using default output directory.")
    return [output_dir] * batch_size


def _tensor_to_pil_image(image_tensor):
    output_image = image_tensor.cpu().numpy()
    image_array = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array[0])


def _build_save_kwargs(
    file_format,
    quality,
    png_compress_level,
    optimize,
    webp_lossless,
    webp_method,
):
    kwargs = {"format": PIL_FORMAT_MAP[file_format]}

    if file_format == "png":
        kwargs["compress_level"] = max(0, min(9, int(png_compress_level)))
    elif file_format == "webp":
        kwargs["quality"] = max(1, min(100, int(quality)))
        kwargs["lossless"] = bool(webp_lossless)
        kwargs["method"] = max(0, min(6, int(webp_method)))
    elif file_format == "jpg":
        kwargs["quality"] = max(1, min(100, int(quality)))
        kwargs["subsampling"] = 0
        if optimize:
            kwargs["optimize"] = True
    elif file_format == "tif" and optimize:
        kwargs["optimize"] = True

    return kwargs


def _save_image_with_fallback(img, output_path, save_kwargs):
    try:
        img.save(output_path, **save_kwargs)
    except OSError as exc:
        if "optimize" in save_kwargs:
            fallback_kwargs = dict(save_kwargs)
            fallback_kwargs.pop("optimize", None)
            print(f"Image save failed with optimize=True, retrying without optimize: {exc}")
            img.save(output_path, **fallback_kwargs)
            return
        raise


def _build_output_filename(filename_mid, numbering, number_prefix):
    if number_prefix:
        return f"{numbering}_{filename_mid}"
    return f"{filename_mid}_{numbering}"


def _find_highest_numeric_value(directory, filename_mid, number_prefix):
    highest_value = -1
    if not os.path.exists(directory):
        return highest_value

    escaped_mid = re.escape(filename_mid)
    if number_prefix:
        pattern = re.compile(rf"^(?P<number>\d+)_{escaped_mid}$")
    else:
        pattern = re.compile(rf"^{escaped_mid}_(?P<number>\d+)$")

    for filename in os.listdir(directory):
        stem, _ = os.path.splitext(filename)
        match = pattern.match(stem)
        if match:
            highest_value = max(highest_value, int(match.group("number")))

    return highest_value


class IO_save_image:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "file_format": (["jpg", "png", "webp", "tif"],),
                "output_path": ("STRING", {"default": "./output", "multiline": False}),
                "filename_mid": ("STRING", {"default": "tmp"}),
            },
            "optional": {
                "number_prefix": ("BOOLEAN", {"default": False, "label_on": "前置编号", "label_off": "后置编号"}),
                "number_digits": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "编号位数，如设置为3则为001格式"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("out_path",)
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Swwan/IO"

    @staticmethod
    def find_highest_numeric_value(directory, filename_mid, number_prefix=False):
        return _find_highest_numeric_value(directory, filename_mid, number_prefix)

    def save_image(self, image, file_format, filename_mid="", output_path="", number_prefix=False, number_digits=5,
                   save_workflow_as_json=False, prompt=None, extra_pnginfo=None):
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]
        output_paths = _resolve_output_paths(output_path, batch_size)

        base_dir = output_paths[0]
        counter = self.find_highest_numeric_value(base_dir, filename_mid, number_prefix) + 1
        absolute_paths = []

        for idx, img_tensor in enumerate(images_list):
            img = _tensor_to_pil_image(img_tensor)
            out_path = output_paths[idx]

            numbering = f"{counter + idx:0{number_digits}d}"
            output_filename = _build_output_filename(filename_mid, numbering, number_prefix)

            extension = FORMAT_EXTENSION_MAP[file_format]
            resolved_image_path = os.path.join(out_path, f"{output_filename}.{extension}")
            _save_image_with_fallback(
                img,
                resolved_image_path,
                _build_save_kwargs(
                    file_format=file_format,
                    quality=80 if file_format == "webp" else 95,
                    png_compress_level=FAST_DEFAULTS["png_compress_level"],
                    optimize=False,
                    webp_lossless=False,
                    webp_method=FAST_DEFAULTS["webp_method"],
                ),
            )
            absolute_paths.append(os.path.abspath(resolved_image_path))

            if save_workflow_as_json:
                try:
                    workflow = (extra_pnginfo or {}).get('workflow')
                    if workflow is not None:
                        json_file_path = os.path.join(out_path, f"{output_filename}.json")
                        with open(json_file_path, 'w') as f:
                            json.dump(workflow, f)
                except Exception as e:
                    print(f"Failed to save workflow JSON: {e}")

        return (absolute_paths, )


class IO_save_image_format:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "file_format": (["jpg", "png", "webp", "tif", "bmp"],),
                "output_path": ("STRING", {"default": "./output", "multiline": False}),
                "filename_mid": ("STRING", {"default": "tmp"}),
            },
            "optional": {
                "number_prefix": ("BOOLEAN", {"default": False, "label_on": "前置编号", "label_off": "后置编号"}),
                "number_digits": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "编号位数，如设置为3则为001格式"}),
                "quality": ("INT", {"default": FAST_DEFAULTS["quality"], "min": 1, "max": 100, "step": 1, "tooltip": "JPEG/WebP 质量。数值越高体积通常越大、保存通常越慢。"}),
                "png_compress_level": ("INT", {"default": FAST_DEFAULTS["png_compress_level"], "min": 0, "max": 9, "step": 1, "tooltip": "PNG 压缩等级。Replicate 场景建议保持较低数值以减少保存耗时。"}),
                "optimize": ("BOOLEAN", {"default": True, "tooltip": "启用 Pillow optimize。主要影响 JPEG/TIFF。若编码器报错，节点会自动回退到不使用 optimize。"}),
                "webp_lossless": ("BOOLEAN", {"default": False, "tooltip": "启用无损 WebP。通常会更慢且文件更大。"}),
                "webp_method": ("INT", {"default": FAST_DEFAULTS["webp_method"], "min": 0, "max": 6, "step": 1, "tooltip": "WebP 编码方法。越高通常压得更小，但更慢。Replicate 场景建议使用 0-2。"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("out_path",)
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Swwan/IO"
    DESCRIPTION = """
Lightweight format-aware image saver for Replicate-oriented workflows.
Keeps only the format parameters that materially affect encode time and file size.
"""

    @staticmethod
    def find_highest_numeric_value(directory, filename_mid, number_prefix=False):
        return _find_highest_numeric_value(directory, filename_mid, number_prefix)

    def save_image(
        self,
        image,
        file_format,
        output_path="",
        filename_mid="",
        number_prefix=False,
        number_digits=5,
        quality=FAST_DEFAULTS["quality"],
        png_compress_level=FAST_DEFAULTS["png_compress_level"],
        optimize=True,
        webp_lossless=False,
        webp_method=FAST_DEFAULTS["webp_method"],
    ):
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]
        output_paths = _resolve_output_paths(output_path, batch_size)

        base_dir = output_paths[0]
        counter = self.find_highest_numeric_value(base_dir, filename_mid, number_prefix) + 1
        absolute_paths = []
        extension = FORMAT_EXTENSION_MAP[file_format]
        save_kwargs = _build_save_kwargs(
            file_format=file_format,
            quality=quality,
            png_compress_level=png_compress_level,
            optimize=optimize,
            webp_lossless=webp_lossless,
            webp_method=webp_method,
        )

        for idx, img_tensor in enumerate(images_list):
            img = _tensor_to_pil_image(img_tensor)
            out_path = output_paths[idx]
            numbering = f"{counter + idx:0{number_digits}d}"
            output_filename = _build_output_filename(filename_mid, numbering, number_prefix)
            resolved_image_path = os.path.join(out_path, f"{output_filename}.{extension}")
            _save_image_with_fallback(img, resolved_image_path, save_kwargs)
            absolute_paths.append(os.path.abspath(resolved_image_path))

        return (absolute_paths, )


NODE_CLASS_MAPPINGS = {
    "IO_save_image": IO_save_image,
    "IO_save_image_format": IO_save_image_format,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IO_save_image": "IO Save Image (Swwan)",
    "IO_save_image_format": "IO Save Image Format (Swwan)",
}
