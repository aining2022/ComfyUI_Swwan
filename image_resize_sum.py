import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple

from comfy.utils import common_upscale

from .main_unit import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#---------------------安全导入

try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  

try:
    from scipy.interpolate import CubicSpline
    REMOVER_AVAILABLE = True  
except ImportError:
    CubicSpline = None
    REMOVER_AVAILABLE = False 

try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  

try:   
    from scipy.ndimage import distance_transform_edt
    REMOVER_AVAILABLE = True  
except ImportError:
    distance_transform_edt = None
    REMOVER_AVAILABLE = False 



#--------------------------------------------------------------------------------------#





class Mask_transform_sum:
    def __init__(self):
        self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "original"}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 400, "step": 1}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "crop_to_mask": ("BOOLEAN", {"default": False}),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "rescale_crop": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {"base_image": ("IMAGE",), "mask": ("MASK",)}
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"
    
    def separate(self, bg_mode, mask_mode="fill", 
                 ignore_threshold=100, opacity=1.0, outline_thickness=1, 
                 smoothness=1, mask_expand=0,
                 expand_width=0, expand_height=0, rescale_crop=1.0,
                 tapered_corners=True, mask_min=0.0, mask_max=1.0,
                 base_image=None, mask=None, crop_to_mask=False, divisible_by=8):
        
        if mask is None:
            if base_image is not None:
                combined_image_tensor = base_image
                empty_mask = torch.zeros_like(base_image[:, :, :, 0])
            else:
                empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
                combined_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (combined_image_tensor, empty_mask)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= ignore_threshold:
                filtered_contours.append(contour)
        
        contours_with_positions = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append((x, y, contour))
        contours_with_positions.sort(key=lambda item: (item[1], item[0]))
        sorted_contours = [item[2] for item in contours_with_positions]
        
        final_mask = np.zeros_like(binary_mask)
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=np.uint8)
        
        for contour in sorted_contours[:8]:
            temp_mask = np.zeros_like(binary_mask)
            
            if mask_mode == "original":
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_image, temp_mask)
            elif mask_mode == "fill":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "fill_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "outline_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline_circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=outline_thickness)
            
            if mask_expand != 0:
                expand_amount = abs(mask_expand)
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=expand_amount)
                else:
                    temp_mask = cv2.erode(temp_mask, kernel, iterations=expand_amount)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_pil = final_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(final_mask_pil)
        
        original_h, original_w = final_mask.shape[:2]
        coords = cv2.findNonZero(final_mask)
        crop_params = None

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            max_expand_left = center_x - 0
            max_expand_right = original_w - center_x
            max_expand_top = center_y - 0
            max_expand_bottom = original_h - center_y
            actual_expand_x = min(expand_width, max_expand_left, max_expand_right)
            actual_expand_y = min(expand_height, max_expand_top, max_expand_bottom)
            x_start = int(round(center_x - (w / 2.0) - actual_expand_x))
            x_end = int(round(center_x + (w / 2.0) + actual_expand_x))
            y_start = int(round(center_y - (h / 2.0) - actual_expand_y))
            y_end = int(round(center_y + (h / 2.0) + actual_expand_y))
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            width = x_end - x_start
            height = y_end - y_start
            if width % 2 != 0:
                if x_end < original_w:
                    x_end += 1
                elif x_start > 0:
                    x_start -= 1
            if height % 2 != 0:
                if y_end < original_h:
                    y_end += 1
                elif y_start > 0:
                    y_start -= 1
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            crop_params = (x_start, y_start, x_end, y_end)
        else:
            crop_params = (0, 0, original_w, original_h)

        if base_image is None:
            base_image_np = np.zeros((original_h, original_w, 3), dtype=np.float32)
        else:
            base_image_np = base_image[0].cpu().numpy() * 255.0
            base_image_np = base_image_np.astype(np.float32)
        
        if crop_to_mask and crop_params is not None:
            x_start, y_start, x_end, y_end = crop_params[:4]
            cropped_final_mask = final_mask[y_start:y_end, x_start:x_end]
            cropped_base_image = base_image_np[y_start:y_end, x_start:x_end].copy()
            
            if rescale_crop != 1.0:
                scaled_w = int(cropped_final_mask.shape[1] * rescale_crop)
                scaled_h = int(cropped_final_mask.shape[0] * rescale_crop)
                cropped_final_mask = cv2.resize(cropped_final_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                cropped_base_image = cv2.resize(cropped_base_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            final_mask = cropped_final_mask
            base_image_np = cropped_base_image
        else:
            if base_image_np.shape[:2] != (original_h, original_w):
                base_image_np = cv2.resize(base_image_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        h, w = base_image_np.shape[:2]
        background = np.zeros((h, w, 3), dtype=np.float32)
        if bg_mode in self.colors:
            background[:] = self.colors[bg_mode]
        elif bg_mode == "image" and base_image is not None:
            background = base_image_np.copy()
        elif bg_mode == "transparent":
            background = np.zeros((h, w, 3), dtype=np.float32)
        
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if bg_mode == "crop_image":
            combined_image = base_image_np.copy()
        elif bg_mode in ["white", "black", "red", "green", "blue", "transparent"]:
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            mask_float = mask_float[:, :, np.newaxis]
            combined_image = mask_float * base_image_np + (1 - mask_float) * background
        elif bg_mode == "image":
            combined_image = background.copy()
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            color = np.array(self.colors["white"], dtype=np.float32)
            for c in range(3):
                combined_image[:, :, c] = (mask_float * (opacity * color[c] + (1 - opacity) * combined_image[:, :, c]) + 
                                         (1 - mask_float) * combined_image[:, :, c])
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)
        
        if divisible_by > 1:
            h, w = combined_image.shape[:2]
            new_h = ((h + divisible_by - 1) // divisible_by) * divisible_by
            new_w = ((w + divisible_by - 1) // divisible_by) * divisible_by
            if new_h != h or new_w != w:
                padded_image = np.zeros((new_h, new_w, 3), dtype=combined_image.dtype)
                padded_image[:h, :w, :] = combined_image
                padded_mask = np.zeros((new_h, new_w), dtype=final_mask.dtype)
                padded_mask[:h, :w] = final_mask
                combined_image = padded_image
                final_mask = padded_mask
        
        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        
        return (combined_image_tensor, final_mask_tensor)


class Image_Resize_sum_data:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "stitch": ("STITCH3",),
            },
            "optional": {
                "image": ("IMAGE",),  # 用于计算缩放因子的图像输入
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "width",          # 排除填充的有效宽度（final_width * scale_factor）
        "height",         # 排除填充的有效高度（final_height * scale_factor）
        "x_offset",       # pad模式时有效图像左上角X坐标 * scale_factor
        "y_offset",       # pad模式时有效图像左上角Y坐标 * scale_factor
        "pad_left",       # 左侧填充像素数 * scale_factor
        "pad_right",      # 右侧填充像素数 * scale_factor
        "pad_top",        # 顶部填充像素数 * scale_factor
        "pad_bottom",     # 底部填充像素数 * scale_factor
        "full_width",     # 包含填充的输出图像实际宽度 * scale_factor
        "full_height",    # 包含填充的输出图像实际高度 * scale_factor
        "scale_factor"    # 计算得到的缩放因子（image宽度 / full_width）
    )
    FUNCTION = "extract_info"
    CATEGORY = "Apt_Preset/image/😺backup"
    DESCRIPTION = """
    从Image_Resize_sum输出的stitch信息中提取关键参数，并根据输入图像自动计算缩放因子：
    1. 缩放因子 = 输入图像宽度 / 原始全宽（包含填充）
    2. 所有输出参数会乘以缩放因子后取整
    3. 若未提供输入图像，缩放因子默认为1.0
    """

    def extract_info(self, stitch: dict, image: torch.Tensor = None) -> Tuple[int, int, int, int, int, int, int, int, int, int, float]:
        # 提取基础信息
        valid_width = stitch.get("final_size", (0, 0))[0]
        valid_height = stitch.get("final_size", (0, 0))[1]
        
        pad_left = stitch.get("pad_info", (0, 0, 0, 0))[0]
        pad_right = stitch.get("pad_info", (0, 0, 0, 0))[1]
        pad_top = stitch.get("pad_info", (0, 0, 0, 0))[2]
        pad_bottom = stitch.get("pad_info", (0, 0, 0, 0))[3]
        
        full_width = valid_width + pad_left + pad_right
        full_height = valid_height + pad_top + pad_bottom
        
        x_offset, y_offset = stitch.get("image_position", (0, 0))

        # 计算缩放因子：image宽度 / full_width（若image存在且full_width不为0）
        if image is not None and full_width > 0:
            # 获取输入图像的宽度（处理批次和单张图像情况）
            if len(image.shape) == 4:  # 批次图像：(B, H, W, C)
                img_width = image.shape[2]
            else:  # 单张图像：(H, W, C)
                img_width = image.shape[1]
            scale_factor = img_width / full_width
        else:
            scale_factor = 1.0  # 默认缩放因子

        # 应用缩放并取整（四舍五入）
        scaled = lambda x: int(round(x * scale_factor))
        
        return (
            scaled(valid_width),
            scaled(valid_height),
            scaled(x_offset),
            scaled(y_offset),
            scaled(pad_left),
            scaled(pad_right),
            scaled(pad_top),
            scaled(pad_bottom),
            scaled(full_width),
            scaled(full_height),
            round(scale_factor, 6)  # 保留6位小数，避免精度问题
        )
  


class Image_Resize_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "keep_proportion": (["resize", "stretch", "pad", "pad_edge", "crop"], ),
                "pad_color": (["black", "white", "red", "green", "blue", "gray"], { "default": "black" }),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                "pad_mask_remove": ("BOOLEAN", {"default": True,}),
            },
            "optional" : {
                "mask": ("MASK",),
                "get_image_size": ("IMAGE",),
                "mask_stack": ("MASK_STACK2",),
            },

        }

    # 增加了remove_pad_mask输出
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH3", "FLOAT", )
    RETURN_NAMES = ("IMAGE", "mask", "stitch",  "scale_factor", )
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    DESCRIPTION = """
    - 输入参数：
    - resize：按比例缩放图像至宽和高的限制范围，保持宽高比，不填充、不裁剪
    - stretch：拉伸图像以完全匹配指定的宽度和高度，保持宽高比、像素扭曲
    - pad：按比例缩放图像后，在目标尺寸内居中放置，用指定颜色填充多余区域
    - pad_edge：与pad类似，但使用图像边缘像素颜色进行填充
    - crop：按目标尺寸比例裁剪原图像，然后缩放到指定尺寸
    - -----------------------  
    - 输出参数：
    - scale_factor：缩放倍率，用于精准还原，可以减少一次缩放导致的模糊
    - remove_pad_mask：移除填充部分的遮罩，保持画布尺寸不变
    """



    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, get_image_size=None, mask=None, mask_stack=None,pad_mask_remove=True):
        if len(image.shape) == 3:
            B, H, W, C = 1, image.shape[0], image.shape[1], image.shape[2]
            original_image = image.unsqueeze(0)
        else:  
            B, H, W, C = image.shape
            original_image = image.clone()
            
        original_H, original_W = H, W

        if width == 0:
            width = W
        if height == 0:
            height = H

        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape
        
        new_width, new_height = width, height
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        crop_x, crop_y, crop_w, crop_h = 0, 0, W, H
        scale_factor = 1.0
        
        processed_mask = mask
        if mask is not None and mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask_stack
            
            separated_result = Mask_transform_sum().separate(  
                bg_mode="crop_image", 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0, 
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image.clone(), 
                mask=mask, 
                crop_to_mask=False,
                divisible_by=1
            )
            processed_mask = separated_result[1]
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                scale_factor = height / H
                new_width = round(W * scale_factor)
                new_height = height
            elif height == 0 and width != 0:
                scale_factor = width / W
                new_width = width
                new_height = round(H * scale_factor)
            elif width != 0 and height != 0:
                scale_factor = min(width / W, height / H)
                new_width = round(W * scale_factor)
                new_height = round(H * scale_factor)

            if keep_proportion.startswith("pad"):
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

        elif keep_proportion == "crop":
            old_aspect = W / H
            new_aspect = width / height
            
            if old_aspect > new_aspect:
                crop_h = H
                crop_w = round(H * new_aspect)
                scale_factor = height / H
            else:
                crop_w = W
                crop_h = round(W / new_aspect)
                scale_factor = width / W
            
            if crop_position == "center":
                crop_x = (W - crop_w) // 2
                crop_y = (H - crop_h) // 2
            elif crop_position == "top":
                crop_x = (W - crop_w) // 2
                crop_y = 0
            elif crop_position == "bottom":
                crop_x = (W - crop_w) // 2
                crop_y = H - crop_h
            elif crop_position == "left":
                crop_x = 0
                crop_y = (H - crop_h) // 2
            elif crop_position == "right":
                crop_x = W - crop_w
                crop_y = (H - crop_h) // 2

        final_width = new_width
        final_height = new_height
        if divisible_by > 1:
            final_width = final_width - (final_width % divisible_by)
            final_height = final_height - (final_height % divisible_by)
            if new_width != 0:
                scale_factor *= (final_width / new_width)
            if new_height != 0:
                scale_factor *= (final_height / new_height)

        out_image = image.clone()
        out_mask = processed_mask.clone() if processed_mask is not None else None
        padding_mask = None

        if keep_proportion == "crop":
            out_image = out_image.narrow(-2, crop_x, crop_w).narrow(-3, crop_y, crop_h)
            if out_mask is not None:
                out_mask = out_mask.narrow(-1, crop_x, crop_w).narrow(-2, crop_y, crop_h)

        out_image = common_upscale(
            out_image.movedim(-1, 1),
            final_width,
            final_height,
            upscale_method,
            crop="disabled"
        ).movedim(1, -1)

        if out_mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(
                    out_mask.unsqueeze(1).repeat(1, 3, 1, 1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(
                    out_mask.unsqueeze(1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).squeeze(1)

        # 保存原始out_mask用于创建remove_pad_mask
        original_out_mask = out_mask.clone() if out_mask is not None else None

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            padded_width = final_width + pad_left + pad_right
            padded_height = final_height + pad_top + pad_bottom
            if divisible_by > 1:
                width_remainder = padded_width % divisible_by
                height_remainder = padded_height % divisible_by
                if width_remainder > 0:
                    extra_width = divisible_by - width_remainder
                    pad_right += extra_width
                    padded_width += extra_width
                if height_remainder > 0:
                    extra_height = divisible_by - height_remainder
                    pad_bottom += extra_height
                    padded_height += extra_height
            
            color_map = {
                "black": "0, 0, 0",
                "white": "255, 255, 255",
                "red": "255, 0, 0",
                "green": "0, 255, 0",
                "blue": "0, 0, 255",
                "gray": "128, 128, 128"
            }
            pad_color_value = color_map[pad_color]
            
            out_image, padding_mask = self.resize_pad(
                out_image,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                0,
                pad_color_value,
                "edge" if keep_proportion == "pad_edge" else "color"
            )
            
            if out_mask is not None:
                out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1, -1)
                out_mask, _ = self.resize_pad(
                    out_mask,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    0,
                    pad_color_value,
                    "edge" if keep_proportion == "pad_edge" else "color"
                )
                out_mask = out_mask[:, :, :, 0]
            else:
                out_mask = torch.ones((B, padded_height, padded_width), dtype=out_image.dtype, device=out_image.device)
                out_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0

        if out_mask is None:
            if keep_proportion != "crop":
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)
            else:
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)

        if padding_mask is not None:
            composite_mask = torch.clamp(padding_mask + out_mask, 0, 1)
        else:
            composite_mask = out_mask.clone()

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            # 获取最终尺寸
            final_padded_height, final_padded_width = composite_mask.shape[1], composite_mask.shape[2]

            remove_pad_mask = torch.zeros_like(composite_mask)
            
            if original_out_mask is not None:
                if original_out_mask.shape[1] != final_height or original_out_mask.shape[2] != final_width:
                    resized_original_mask = common_upscale(
                        original_out_mask.unsqueeze(1),
                        final_width,
                        final_height,
                        upscale_method,
                        crop="disabled"
                    ).squeeze(1)
                else:
                    resized_original_mask = original_out_mask
        
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = resized_original_mask
            else:
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0
        else:
            remove_pad_mask = composite_mask.clone()

        stitch_info = {
            "original_image": original_image,
            "original_shape": (original_H, original_W),
            "resized_shape": (out_image.shape[1], out_image.shape[2]),
            "crop_position": (crop_x, crop_y),
            "crop_size": (crop_w, crop_h),
            "pad_info": (pad_left, pad_right, pad_top, pad_bottom),
            "keep_proportion": keep_proportion,
            "upscale_method": upscale_method,
            "scale_factor": scale_factor,
            "final_size": (final_width, final_height),
            "image_position": (pad_left, pad_top) if keep_proportion.startswith("pad") else (0, 0),
            "has_input_mask": mask is not None,
            "original_mask": mask.clone() if mask is not None else None
        }
        
        scale_factor = 1/scale_factor

        if pad_mask_remove:
           Fina_mask =  remove_pad_mask.cpu()
        else:
           Fina_mask =  composite_mask.cpu()

        return (out_image.cpu(), Fina_mask, stitch_info, scale_factor, )


    def resize_pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape

        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        bg_color = [int(x.strip()) / 255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # 新增逻辑：判断是否需要跳过缩放
        should_skip_resize = False
        if target_width is not None and target_height is not None:
            # 判断长边是否已经等于目标尺寸
            current_long_side = max(W, H)
            target_long_side = max(target_width, target_height)
            if current_long_side == target_long_side:
                should_skip_resize = True

        if not should_skip_resize and target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "bilinear", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            pad_left = (target_width - W) // 2
            pad_right = target_width - W - pad_left
            pad_top = (target_height - H) // 2
            pad_bottom = target_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

        padded_width = W + pad_left + pad_right
        padded_height = H + pad_top + pad_bottom

        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]

                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            else:
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        padding_mask = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        for m in range(B):
            padding_mask[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, padding_mask)



class Image_Resize_sum_restore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resized_image": ("IMAGE",),
                "stitch": ("STITCH3",),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "pad_crop_no_scale": ("BOOLEAN", {"default": False, }),
            },
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("restored_image", "restored_mask", "original_image")
    FUNCTION = "restore"

    DESCRIPTION = """
    - 新增参数：
    - pad_crop_no_scale：布尔值，True时无缩放直接裁切填充部分
      （计算输入图像与Image_Resize_sum输出图像的尺寸倍数，对填充部分做同倍数裁切）
    """

    def restore(self, resized_image, stitch, upscale_method="bicubic", pad_crop_no_scale=False):
        # 从stitch中提取关键信息
        original_h, original_w = stitch["original_shape"]
        pad_left, pad_right, pad_top, pad_bottom = stitch["pad_info"]
        keep_proportion = stitch["keep_proportion"]
        final_width, final_height = stitch["final_size"]  # Image_Resize_sum输出的有效区域尺寸
        original_mask = stitch.get("original_mask")
        has_input_mask = stitch.get("has_input_mask", False)
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        original_resized_shape = stitch["resized_shape"]  # Image_Resize_sum输出的完整尺寸（含填充）
        
        # 获取原始图像
        original_image = stitch.get("original_image", None)
        
        # 获取当前输入图像的尺寸
        current_b, current_h, current_w, current_c = resized_image.shape

        # 计算尺寸倍数：当前输入图像尺寸 / Image_Resize_sum输出的完整尺寸
        scale_w = current_w / original_resized_shape[1] if original_resized_shape[1] != 0 else 1.0
        scale_h = current_h / original_resized_shape[0] if original_resized_shape[0] != 0 else 1.0

        if pad_crop_no_scale:
            # 无缩放直接裁切模式：按倍数计算实际填充区域并裁切
            if keep_proportion.startswith("pad"):
                # 计算当前图像中实际的填充区域（按尺寸倍数缩放）
                current_pad_left = int(round(pad_left * scale_w))
                current_pad_right = int(round(pad_right * scale_w))
                current_pad_top = int(round(pad_top * scale_h))
                current_pad_bottom = int(round(pad_bottom * scale_h))
                
                # 安全裁切：确保裁切范围在当前图像内
                crop_left = max(0, current_pad_left)
                crop_right = max(0, current_w - current_pad_right)
                crop_top = max(0, current_pad_top)
                crop_bottom = max(0, current_h - current_pad_bottom)
                
                # 裁切填充部分，得到有效区域（不缩放）
                valid_image = resized_image[:, crop_top:crop_bottom, crop_left:crop_right, :]
                
                # 直接使用裁切后的有效区域作为还原图像（保持原有尺寸，仅移除填充）
                restored_image = valid_image
            elif keep_proportion == "crop":
                # crop模式下：按原始裁剪比例裁切当前图像（不缩放）
                original_cropped_ratio = crop_w / crop_h if crop_h != 0 else 1.0
                current_ratio = current_w / current_h if current_h != 0 else 1.0
                
                if abs(current_ratio - original_cropped_ratio) > 1e-6:
                    if current_ratio > original_cropped_ratio:
                        # 当前图像更宽，按高度裁切宽度
                        target_w = int(round(current_h * original_cropped_ratio))
                        crop_left = (current_w - target_w) // 2
                        valid_image = resized_image[:, :, crop_left:crop_left+target_w, :]
                    else:
                        # 当前图像更高，按宽度裁切高度
                        target_h = int(round(current_w / original_cropped_ratio))
                        crop_top = (current_h - target_h) // 2
                        valid_image = resized_image[:, crop_top:crop_top+target_h, :, :]
                else:
                    valid_image = resized_image
                
                # 放回原始图像位置（不缩放）
                if stitch.get("original_image") is not None:
                    restored_image = stitch["original_image"].clone()
                else:
                    restored_image = torch.zeros(
                        (current_b, original_h, original_w, current_c),
                        dtype=resized_image.dtype,
                        device=resized_image.device
                    )
                
                # 调整有效图像尺寸以匹配原始裁剪区域（仅调整尺寸，不缩放内容）
                valid_image_resized = common_upscale(
                    valid_image.movedim(-1, 1),
                    crop_w, crop_h,
                    "nearest-exact",  # 直接调整尺寸，不插值
                    crop="disabled"
                ).movedim(1, -1)
                
                restored_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = valid_image_resized
            else:
                # resize/stretch模式：直接返回当前图像（不缩放）
                restored_image = resized_image
        else:
            # 原有逻辑：带缩放的还原
            if keep_proportion.startswith("pad"):
                # 计算原始有效区域在填充后图像中的占比（用于处理比例变化）
                original_padded_w = final_width + pad_left + pad_right
                original_padded_h = final_height + pad_top + pad_bottom
                
                # 计算当前图像中有效区域的实际位置和尺寸（考虑比例变化）
                scale_w = current_w / original_padded_w if original_padded_w != 0 else 1.0
                scale_h = current_h / original_padded_h if original_padded_h != 0 else 1.0
                
                current_pad_left = int(round(pad_left * scale_w))
                current_pad_top = int(round(pad_top * scale_h))
                current_valid_w = int(round(final_width * scale_w))
                current_valid_h = int(round(final_height * scale_h))
                
                # 安全裁剪
                valid_left = max(0, current_pad_left)
                valid_right = min(current_w, current_pad_left + current_valid_w)
                valid_top = max(0, current_pad_top)
                valid_bottom = min(current_h, current_pad_top + current_valid_h)
                
                valid_image = resized_image[:, valid_top:valid_bottom, valid_left:valid_right, :]
                
                # 缩放至原始尺寸
                restored_image = common_upscale(
                    valid_image.movedim(-1, 1),
                    original_w, original_h,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)

            elif keep_proportion == "crop":
                # 处理crop模式的比例适配
                original_cropped_ratio = crop_w / crop_h if crop_h != 0 else 1.0
                current_ratio = current_w / current_h if current_h != 0 else 1.0
                
                if abs(current_ratio - original_cropped_ratio) > 1e-6:
                    if current_ratio > original_cropped_ratio:
                        target_w = int(round(current_h * original_cropped_ratio))
                        crop_left = (current_w - target_w) // 2
                        crop_right = current_w - target_w - crop_left
                        valid_image = resized_image[:, :, crop_left:current_w - crop_right, :]
                    else:
                        target_h = int(round(current_w / original_cropped_ratio))
                        crop_top = (current_h - target_h) // 2
                        crop_bottom = current_h - target_h - crop_top
                        valid_image = resized_image[:, crop_top:current_h - crop_bottom, :, :]
                else:
                    valid_image = resized_image
                
                # 缩放至原始裁剪区域尺寸
                crop_restored = common_upscale(
                    valid_image.movedim(-1, 1),
                    crop_w, crop_h,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)
                
                # 放回原始图像位置
                if stitch.get("original_image") is not None:
                    restored_image = stitch["original_image"].clone()
                else:
                    restored_image = torch.zeros(
                        (current_b, original_h, original_w, current_c),
                        dtype=resized_image.dtype,
                        device=resized_image.device
                    )
                restored_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = crop_restored

            else:  # resize/stretch模式
                # 直接按原始尺寸比例缩放
                restored_image = common_upscale(
                    resized_image.movedim(-1, 1),
                    original_w, original_h,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)

        # 处理mask（无缩放模式下同步裁切mask）
        if pad_crop_no_scale and original_mask is not None and has_input_mask:
            # 对原始mask按相同倍数裁切填充区域
            mask_h, mask_w = original_mask.shape[1], original_mask.shape[2]
            mask_scale_w = mask_w / original_resized_shape[1] if original_resized_shape[1] != 0 else 1.0
            mask_scale_h = mask_h / original_resized_shape[0] if original_resized_shape[0] != 0 else 1.0
            
            mask_pad_left = int(round(pad_left * mask_scale_w))
            mask_pad_right = int(round(pad_right * mask_scale_w))
            mask_pad_top = int(round(pad_top * mask_scale_h))
            mask_pad_bottom = int(round(pad_bottom * mask_scale_h))
            
            mask_crop_left = max(0, mask_pad_left)
            mask_crop_right = max(0, mask_w - mask_pad_right)
            mask_crop_top = max(0, mask_pad_top)
            mask_crop_bottom = max(0, mask_h - mask_pad_bottom)
            
            restored_mask = original_mask[:, mask_crop_top:mask_crop_bottom, mask_crop_left:mask_crop_right]
        else:
            restored_mask = original_mask if (original_mask is not None and has_input_mask) else (
                torch.zeros((current_b, original_h, original_w), dtype=torch.float32, device=resized_image.device)
            )
       
        # 确保输出为PIL图像（保持原有逻辑）
        if isinstance(restored_image, torch.Tensor):
            restored_image = convert_pil_image(restored_image)
        
        # 处理原始图像输出
        if original_image is not None:
            output_original_image = original_image
        else:
            output_original_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)

        return (restored_image.cpu(), restored_mask.cpu(), output_original_image.cpu())



NODE_CLASS_MAPPINGS = {
    "Mask_transform_sum": Mask_transform_sum,
    "Image_Resize_sum": Image_Resize_sum,
    "Image_Resize_sum_restore": Image_Resize_sum_restore,
    "Image_Resize_sum_data": Image_Resize_sum_data,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask_transform_sum": "Mask Transform Sum",
    "Image_Resize_sum": "Image Resize Sum",
    "Image_Resize_sum_restore": "Image Resize Sum Restore",
    "Image_Resize_sum_data": "Image Resize Sum Data",
}
