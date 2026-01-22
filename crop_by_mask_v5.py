import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management

from .layerstyle_utils import log, tensor2pil, pil2tensor, mask2image, image2mask
from .layerstyle_utils import min_bounding_rect, max_inscribed_rect, mask_area
from .layerstyle_utils import num_round_up_to_multiple, draw_rect


class CropByMaskV5:
    """
    CropByMask V5 - 批处理增强版智能裁剪节点

    相比 V4 新增功能：
    - batch_mode: 批处理模式，支持直接处理视频批次
      * single_frame: 单帧模式，每帧独立计算 crop_box
      * batch_first_reuse: 批处理模式，首帧计算 crop_box，后续帧自动复用
    """

    def __init__(self):
        self.NODE_NAME = 'CropByMask V5'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        batch_mode_list = ['single_frame', 'batch_first_reuse']
        reserve_mode_list = ['absolute', 'ratio']
        return {
            "required": {
                "detect": (detect_mode,),
                "reserve_mode": (reserve_mode_list,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "top_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "bottom_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "left_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "right_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "reserve_max": ("INT", {"default": 100, "min": 0, "max": 1024, "step": 1}),
                "round_to_multiple": (multiple_list,),
                "batch_mode": (batch_mode_list,),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOX", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v5'
    CATEGORY = 'Swwan/Image'
    DESCRIPTION = """CropByMask V5 - 批处理智能裁剪节点

批处理模式:
- single_frame: 单帧模式，逐帧独立计算（兼容 V4）
- batch_first_reuse: 批处理模式，首帧计算 crop_box，后续帧自动复用

Reserve 模式:
- absolute: 绝对值模式，使用固定像素值 (top/bottom/left/right_reserve)
- ratio: 比例模式，基于 mask 尺寸计算 reserve
  * 左右 reserve = mask宽度 × ratio，上下 reserve = mask高度 × ratio
  * reserve_max 限制最大扩展像素（防止过度扩展）
  * 推荐比例：0.3(紧凑), 0.5(标准), 0.8(宽松)

使用说明:
1. 直接接收视频批次输入（无需 ImageList 转换）
2. batch_first_reuse 模式下，自动处理首帧和后续帧
3. ratio 模式适合不同尺寸的 mask（如视频中人脸大小变化）
4. device 参数支持 CPU/GPU 选择"""

    def _gaussian_blur_tensor(self, mask_tensor, blur_radius, processing_device):
        """PyTorch 实现的高斯模糊，支持 GPU 加速"""
        if blur_radius <= 0:
            return mask_tensor

        mask_tensor = mask_tensor.to(processing_device)
        mask_4d = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        kernel_size = int(blur_radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = blur_radius / 3.0

        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                         dtype=torch.float32, device=processing_device)
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2
        blurred = F.conv2d(mask_4d, gaussian_2d, padding=padding)
        return blurred.squeeze(0).squeeze(0)

    def _calculate_crop_box(self, mask_pil, detect, reserve_mode,
                            top_reserve, bottom_reserve, left_reserve, right_reserve,
                            top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                            reserve_max, round_to_multiple,
                            canvas_width, canvas_height, device):
        """计算 crop_box"""

        # 选择设备
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        # GPU 加速的高斯模糊
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)
        blurred_tensor = self._gaussian_blur_tensor(mask_tensor, 20, processing_device)

        # 转回 PIL 进行检测（OpenCV 只能在 CPU）
        blurred_np = (blurred_tensor.cpu().numpy() * 255).astype(np.uint8)
        bluredmask = Image.fromarray(blurred_np, mode='L')

        x = 0
        y = 0
        w = 0
        h = 0

        if detect == "min_bounding_rect":
            (x, y, w, h) = min_bounding_rect(bluredmask)
        elif detect == "max_inscribed_rect":
            (x, y, w, h) = max_inscribed_rect(bluredmask)
        else:  # mask_area
            (x, y, w, h) = mask_area(mask_pil)

        # 根据 reserve_mode 计算实际的 reserve 值
        if reserve_mode == "ratio":
            # 比例模式：基于 mask 尺寸计算
            mask_width = w
            mask_height = h

            # 计算绝对像素值（左右基于宽度，上下基于高度）
            left_reserve_abs = min(mask_width * left_reserve_ratio, reserve_max)
            right_reserve_abs = min(mask_width * right_reserve_ratio, reserve_max)
            top_reserve_abs = min(mask_height * top_reserve_ratio, reserve_max)
            bottom_reserve_abs = min(mask_height * bottom_reserve_ratio, reserve_max)
        else:
            # 绝对值模式：直接使用像素值
            left_reserve_abs = left_reserve
            right_reserve_abs = right_reserve
            top_reserve_abs = top_reserve
            bottom_reserve_abs = bottom_reserve

        # 应用边距保留（使用计算后的绝对值）
        x1 = max(0, x - left_reserve_abs)
        y1 = max(0, y - top_reserve_abs)
        x2 = min(canvas_width, x + w + right_reserve_abs)
        y2 = min(canvas_height, y + h + bottom_reserve_abs)

        # 调整到指定倍数
        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            width = num_round_up_to_multiple(x2 - x1, multiple)
            height = num_round_up_to_multiple(y2 - y1, multiple)

            # 居中调整
            x1 = x1 - (width - (x2 - x1)) // 2
            y1 = y1 - (height - (y2 - y1)) // 2

            # 确保不超出边界
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            x2 = x1 + width
            y2 = y1 + height

            # 如果超出右边或底部边界，向左/上调整
            if x2 > canvas_width:
                x2 = canvas_width
                x1 = max(0, x2 - width)
            if y2 > canvas_height:
                y2 = canvas_height
                y1 = max(0, y2 - height)

        # 确保返回整数（避免 numpy.float64 导致的类型错误）
        return (int(x1), int(y1), int(x2), int(y2)), (int(x), int(y), int(w), int(h))

    def crop_by_mask_v5(self, detect, reserve_mode,
                        top_reserve, bottom_reserve, left_reserve, right_reserve,
                        top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                        reserve_max, round_to_multiple,
                        batch_mode="single_frame", device="CPU",
                        image=None, mask_image=None, crop_box=None):

        # 验证输入
        if mask_image is None and crop_box is None:
            raise ValueError("Either mask_image or crop_box must be provided")

        if image is None and mask_image is None:
            raise ValueError("At least one of image or mask_image must be provided")

        # 确定画布尺寸
        if image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
            batch_size = image.shape[0]
        elif mask_image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
            batch_size = 1
        else:
            raise ValueError("Cannot determine canvas size")

        ret_images = []
        ret_masks = []
        effective_crop_box = None
        preview_image = None

        # 处理遮罩图像（使用第一帧的 mask）
        if mask_image is not None:
            mask_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('L')
            preview_image = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
        else:
            mask_pil = Image.new('L', (canvas_width, canvas_height), 255)
            preview_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # 确定使用的 crop_box（批处理模式下只计算一次）
        if crop_box is not None:
            # 优先使用输入的 crop_box
            effective_crop_box = tuple(crop_box)
            log(f"{self.NODE_NAME}: Using input crop_box: {effective_crop_box}")
        else:
            # 计算新的 crop_box（只计算一次，应用到所有帧）
            effective_crop_box, detected_rect = self._calculate_crop_box(
                mask_pil, detect, reserve_mode,
                top_reserve, bottom_reserve, left_reserve, right_reserve,
                top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                reserve_max, round_to_multiple,
                canvas_width, canvas_height, device
            )
            log(f"{self.NODE_NAME}: Calculated crop_box: {effective_crop_box}")

            # 在预览图上绘制检测到的区域（红色）
            if detected_rect:
                x, y, w, h = detected_rect
                preview_image = draw_rect(preview_image, x, y, w, h, line_color="#FF0000",
                                          line_width=max(1, (w + h) // 100))

        # 在预览图上绘制最终裁剪框（绿色）
        preview_image = draw_rect(preview_image, effective_crop_box[0], effective_crop_box[1],
                                  effective_crop_box[2] - effective_crop_box[0],
                                  effective_crop_box[3] - effective_crop_box[1],
                                  line_color="#00FF00",
                                  line_width=max(1, (effective_crop_box[2] - effective_crop_box[0] +
                                                     effective_crop_box[3] - effective_crop_box[1]) // 200))

        # 批处理模式：应用同一个 crop_box 到所有帧
        if image is not None:
            if batch_mode == "batch_first_reuse":
                log(f"{self.NODE_NAME}: Batch mode - processing {batch_size} frames with same crop_box")

            # 裁剪 mask（首帧mask复用到所有帧）
            cropped_mask_pil = mask_pil.crop(effective_crop_box)

            for i in range(batch_size):
                img_pil = tensor2pil(torch.unsqueeze(image[i], 0)).convert('RGB')
                cropped_img = img_pil.crop(effective_crop_box)
                ret_images.append(pil2tensor(cropped_img))

                # 每帧都添加对应的 mask（复用同一个裁剪后的 mask）
                ret_masks.append(pil2tensor(cropped_mask_pil.convert('RGB')))
        else:
            # 只有 mask_image 输入的情况
            cropped_mask = mask_pil.crop(effective_crop_box)
            ret_images.append(pil2tensor(cropped_mask.convert('RGB')))
            ret_masks.append(pil2tensor(cropped_mask.convert('RGB')))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            list(effective_crop_box),
            pil2tensor(preview_image),
        )


NODE_CLASS_MAPPINGS = {
    "SwwanCropByMaskV5": CropByMaskV5
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwwanCropByMaskV5": "Crop By Mask V5 (Batch)"
}
