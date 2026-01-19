import os
import json
import re
import torch
import numpy as np
from PIL import Image
import folder_paths


class IO_save_image:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "file_format": (["png", "webp", "jpg", "tif"],),
                "output_path": ("STRING", {"default": "./output/Swwan", "multiline": False}),
                "filename_mid": ("STRING", {"default": "Swwan"}),
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
    def find_highest_numeric_value(directory, filename_mid):
        highest_value = -1
        if not os.path.exists(directory):
            return highest_value
        for filename in os.listdir(directory):
            if filename.startswith(filename_mid):
                try:
                    numeric_part = filename[len(filename_mid):]
                    numeric_str = re.search(r'\d+', numeric_part).group()
                    numeric_value = int(numeric_str)
                    if numeric_value > highest_value:
                        highest_value = numeric_value
                except (ValueError, AttributeError):
                    continue
        return highest_value

    def save_image(self, image, file_format, filename_mid="Swwan", output_path="", number_prefix=False, number_digits=5,
                   save_workflow_as_json=False, prompt=None, extra_pnginfo=None):
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]
        output_dir = folder_paths.get_output_directory()
        output_paths = []

        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            output_paths = [output_path] * batch_size
        elif isinstance(output_path, list) and len(output_path) == batch_size:
            for path in output_path:
                os.makedirs(path, exist_ok=True)
            output_paths = output_path
        else:
            print("Invalid output_path format. Using default output directory.")
            output_paths = [output_dir] * batch_size

        base_dir = output_paths[0]
        counter = self.find_highest_numeric_value(base_dir, filename_mid) + 1
        absolute_paths = []

        for idx, img_tensor in enumerate(images_list):
            output_image = img_tensor.cpu().numpy()
            img_np = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np[0])
            out_path = output_paths[idx]

            numbering = f"{counter + idx:0{number_digits}d}"
            if number_prefix:
                output_filename = f"{numbering}_{filename_mid}"
            else:
                output_filename = f"{filename_mid}_{numbering}"

            resolved_image_path = os.path.join(out_path, f"{output_filename}.{file_format}")
            img_params = {
                'png': {'compress_level': 4},
                'webp': {'method': 6, 'lossless': False, 'quality': 80},
                'jpg': {'quality': 95, 'format': 'JPEG'},
                'tif': {'format': 'TIFF'}
            }
            img.save(resolved_image_path, **img_params[file_format])
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


NODE_CLASS_MAPPINGS = {
    "IO_save_image": IO_save_image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IO_save_image": "IO Save Image (Swwan)",
}
