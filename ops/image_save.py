# SPDX-License-Identifier: MIT
# Original Swwan saving helpers, shared with the new saver.
import os
import re
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args
FORMAT_EXTENSION_MAP = {"png":"png", "webp":"webp", "jpg":"jpg", "tif":"tif", "bmp":"bmp"}
PIL_FORMAT_MAP = {"png":"PNG", "webp":"WEBP", "jpg":"JPEG", "tif":"TIFF", "bmp":"BMP"}
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


def _legacy_output_folder(folder, root):
    """KJ compatibility: historical default means root; custom relative folders join root."""
    return root if folder in ('', 'output', './output') else os.path.join(root, folder)
