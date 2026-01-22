import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management

from .layerstyle_utils import log, tensor2pil, pil2tensor, mask2image, image2mask
from .layerstyle_utils import min_bounding_rect, max_inscribed_rect, mask_area
from .layerstyle_utils import num_round_up_to_multiple, draw_rect


class CropByMaskV4:
    """
    CropByMask V4 - 增强版智能裁剪节点

    相比 V3 新增功能：
    - reuse_crop_box: 复用首次计算的 crop_box，适合视频批量处理
    - device: 支持 CPU/GPU 选择
    """

    def __init__(self):
        self.NODE_NAME = 'CropByMask V4'
        self._cached_crop_box = None  # 缓存 crop_box

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
                "reuse_crop_box": ("BOOLEAN", {"default": False}),
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
    FUNCTION = 'crop_by_mask_v4'
    CATEGORY = 'RunningHub/LayerUtility'
    DESCRIPTION = """CropByMask V4 - 智能裁剪节点

新增参数:
- reuse_crop_box: 复用首次计算的 crop_box，视频处理时只计算一次，后续帧直接复用
- device: 选择 CPU 或 GPU 进行处理"""

    def _gaussian_blur_tensor(self, mask_tensor, blur_radius, processing_device):
        """
        PyTorch 实现的高斯模糊，支持 GPU 加速

        Args:
            mask_tensor: [H, W] 的 tensor
            blur_radius: 模糊半径
            processing_device: 计算设备

        Returns:
            模糊后的 tensor [H, W]
        """
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

    def _calculate_crop_box(self, mask_pil, detect, top_reserve, bottom_reserve,
                            left_reserve, right_reserve, round_to_multiple,
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

        # 应用边距保留
        x1 = max(0, x - left_reserve)
        y1 = max(0, y - top_reserve)
        x2 = min(canvas_width, x + w + right_reserve)
        y2 = min(canvas_height, y + h + bottom_reserve)

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

        return (x1, y1, x2, y2), (x, y, w, h)

    def crop_by_mask_v4(self, detect, top_reserve, bottom_reserve,
                        left_reserve, right_reserve, round_to_multiple,
                        reuse_crop_box=False, device="CPU",
                        image=None, mask_image=None, crop_box=None):

        # 验证输入
        if mask_image is None and crop_box is None and not (reuse_crop_box and self._cached_crop_box is not None):
            raise ValueError("Either mask_image or crop_box must be provided (or enable reuse_crop_box with cached value)")

        if image is None and mask_image is None:
            raise ValueError("At least one of image or mask_image must be provided")

        # 确定画布尺寸
        if image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
        elif mask_image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
        else:
            raise ValueError("Cannot determine canvas size")

        ret_images = []
        ret_masks = []
        detected_rect = None

        # 处理遮罩图像
        if mask_image is not None:
            mask_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('L')
            preview_image = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
        else:
            mask_pil = Image.new('L', (canvas_width, canvas_height), 255)
            preview_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # 确定使用的 crop_box
        effective_crop_box = None

        if crop_box is not None:
            # 优先使用输入的 crop_box
            effective_crop_box = tuple(crop_box)
            self._cached_crop_box = effective_crop_box
            log(f"{self.NODE_NAME}: Using input crop_box: {effective_crop_box}")
        elif reuse_crop_box and self._cached_crop_box is not None:
            # 使用缓存的 crop_box
            effective_crop_box = self._cached_crop_box
            log(f"{self.NODE_NAME}: Reusing cached crop_box: {effective_crop_box}")
        else:
            # 计算新的 crop_box
            effective_crop_box, detected_rect = self._calculate_crop_box(
                mask_pil, detect, top_reserve, bottom_reserve,
                left_reserve, right_reserve, round_to_multiple,
                canvas_width, canvas_height, device
            )
            self._cached_crop_box = effective_crop_box
            log(f"{self.NODE_NAME}: Calculated new crop_box: {effective_crop_box}")

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

        # 裁剪图像
        if image is not None:
            for i in range(image.shape[0]):
                img_pil = tensor2pil(torch.unsqueeze(image[i], 0)).convert('RGB')
                cropped_img = img_pil.crop(effective_crop_box)
                ret_images.append(pil2tensor(cropped_img))
        else:
            cropped_mask = mask_pil.crop(effective_crop_box)
            ret_images.append(pil2tensor(cropped_mask.convert('RGB')))

        # 裁剪遮罩
        cropped_mask = mask_pil.crop(effective_crop_box)
        ret_masks.append(pil2tensor(cropped_mask.convert('RGB')))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            list(effective_crop_box),
            pil2tensor(preview_image),
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V4": CropByMaskV4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V4": "LayerUtility: CropByMask V4"
}
