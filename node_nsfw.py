import cv2
import numpy as np
from .r_nudenet.nudenet import NudeDetector
import os
import torch
import folder_paths as comfy_paths
from folder_paths import models_dir
from typing import Union, List
import json
import logging

logger = logging.getLogger(__file__)

comfy_paths.folder_names_and_paths["nsfw"] = ([os.path.join(models_dir, "nsfw")], {".pt",".onnx"})


def tensor2np(tensor: torch.Tensor):
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        if len(img_np) == 0:
            return torch.tensor([])
        return torch.cat([np2tensor(img) for img in img_np], dim=0)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


class DetectorForNSFW:

    def __init__(self) -> None:
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_size":([640, 320], {"default": 320}),
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
            "optional": {
                "model_name": (comfy_paths.get_filename_list("nsfw") + [""], {"default": ""}),
                "alternative_image": ("IMAGE",),
                "buttocks_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_breast_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "anus_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "male_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("output_image", "detect_result", "filtered_image")
    FUNCTION = "filter_exposure"

    CATEGORY = "Swwan/filter"

    all_labels = [
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    ]

    def filter_exposure(self, image, model_name=None, detect_size=320, provider="CPU", alternative_image=None, **kwargs):
        if self.model is None:
            self.init_model(model_name, detect_size, provider)

        if alternative_image is not None:
            alternative_image = tensor2np(alternative_image)
            if not isinstance(alternative_image, List):
                alternative_image = [alternative_image]

        images = tensor2np(image)
        if not isinstance(images, List):
            images = [images]

        results, result_info, filtered_results = [],[],[]
        for i, img in enumerate(images):
            detect_result = self.model.detect(img)

            logger.debug(f"nudenet detect result:{detect_result}")
            detected_results = []
            for item in detect_result:
                label = item['class']
                score = item['score']
                confidence_level = kwargs.get(label.lower())
                if label.lower() in kwargs and score > confidence_level:
                    detected_results.append(item)
            info = {"detect_result":detect_result}
            if len(detected_results) == 0:
                results.append(img)
                info["nsfw"] = False
                filtered_results.append(img)
            else:
                if alternative_image is not None:
                    placeholder_image = alternative_image[i] if len(alternative_image) == len(images) else alternative_image[0]
                else:
                    placeholder_image = np.ones_like(img) * 255
                results.append(placeholder_image)
                info["nsfw"] = True

            result_info.append(info)

        result_tensor = np2tensor(results)
        filtered_tensor = np2tensor(filtered_results)
        result_info = json.dumps(result_info)
        return (result_tensor, result_info, filtered_tensor)

    def init_model(self, model_name, detect_size, provider):
        model_path = comfy_paths.get_full_path("nsfw", model_name) if model_name else None
        self.model = NudeDetector(model_path=model_path, providers=[provider + 'ExecutionProvider',], inference_resolution=detect_size)


class DetectorForNSFWV2:
    """NSFW Detector V2 - 支持检测到NSFW内容时抛出异常终止工作流"""

    def __init__(self) -> None:
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_size":([640, 320], {"default": 320}),
                "provider": (["CPU", "CUDA", "ROCM"], ),
                "raise_exception": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_name": (comfy_paths.get_filename_list("nsfw") + [""], {"default": ""}),
                "exception_message": ("STRING", {"default": "NSFW content detected, workflow terminated."}),
                "buttocks_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_breast_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "anus_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "male_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "is_nsfw", "detect_result")
    FUNCTION = "detect_nsfw"

    CATEGORY = "Swwan/filter"

    def detect_nsfw(self, image, detect_size=320, provider="CPU", raise_exception=False,
                    model_name=None, exception_message="NSFW content detected, workflow terminated.", **kwargs):
        if self.model is None:
            self.init_model(model_name, detect_size, provider)

        images = tensor2np(image)
        if not isinstance(images, List):
            images = [images]

        is_nsfw = False
        result_info = []

        for i, img in enumerate(images):
            detect_result = self.model.detect(img)
            logger.debug(f"nudenet detect result:{detect_result}")

            detected_results = []
            for item in detect_result:
                label = item['class']
                score = item['score']
                confidence_level = kwargs.get(label.lower())
                if label.lower() in kwargs and score > confidence_level:
                    detected_results.append(item)

            info = {"detect_result": detect_result, "nsfw": len(detected_results) > 0}
            result_info.append(info)

            if len(detected_results) > 0:
                is_nsfw = True

        if is_nsfw and raise_exception:
            raise Exception(exception_message)

        return (image, is_nsfw, json.dumps(result_info))

    def init_model(self, model_name, detect_size, provider):
        model_path = comfy_paths.get_full_path("nsfw", model_name) if model_name else None
        self.model = NudeDetector(model_path=model_path, providers=[provider + 'ExecutionProvider',], inference_resolution=detect_size)


class RaiseExceptionOnTrue:
    """当输入为True时抛出自定义异常，用于终止工作流节省计算成本"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": False}),
                "exception_message": ("STRING", {"default": "Workflow terminated by condition."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "check_and_raise"

    CATEGORY = "Swwan/utils"

    def check_and_raise(self, condition, exception_message="Workflow terminated by condition."):
        if condition:
            raise Exception(exception_message)
        return (condition,)


NODE_CLASS_MAPPINGS = {
    "nsfwDetector": DetectorForNSFW,
    "nsfwDetectorV2": DetectorForNSFWV2,
    "raiseExceptionOnTrue": RaiseExceptionOnTrue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "nsfwDetector": "NSFW Detector",
    "nsfwDetectorV2": "NSFW Detector V2",
    "raiseExceptionOnTrue": "Raise Exception On True",
}
