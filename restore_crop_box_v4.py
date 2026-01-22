import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management
from .layerstyle_utils import log


class RestoreCropBoxV4:
    """
    RestoreCropBox V4 - PyTorch 批处理优化版

    相比 V3 改进：
    - 完全 PyTorch 实现，无 PIL 转换（10-100x 性能提升）
    - 批量处理所有帧，GPU 加速友好
    - 直接输出 IMAGE batch，无需额外转换
    """

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox V4'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "crop_box": ("BOX",),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "device": (["CPU", "GPU"], {"default": "GPU"}),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'Swwan/Image'
    DESCRIPTION = """RestoreCropBox V4 - PyTorch 批处理优化版

性能优化:
- 完全 PyTorch 实现，避免 Tensor ↔ PIL 转换
- 批量处理所有帧，10-100x 性能提升
- GPU 加速友好
- 直接输出 IMAGE batch，无需 ImageListToImageBatch

推荐配置:
- device: GPU（批量处理时性能提升明显）
- feathering: 30-60（中等分辨率）

注意:
- 输入的 background_image 和 croped_image 必须批次大小相同
- 输出直接可用于 VideoCombine"""

    def _create_feathered_mask_batch(self, mask_batch, feathering, device):
        """
        批量应用羽化效果（PyTorch 实现）

        Args:
            mask_batch: [B, H, W] 或 [B, H, W, 1] 的 tensor
            feathering: 羽化半径
            device: 处理设备

        Returns:
            羽化后的 mask [B, H, W]
        """
        if feathering <= 0:
            return mask_batch.squeeze(-1) if mask_batch.dim() == 4 else mask_batch

        # 确保是 [B, H, W] 格式
        if mask_batch.dim() == 4:
            mask_batch = mask_batch.squeeze(-1)

        # 移动到指定设备
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        mask_batch = mask_batch.to(processing_device)

        # [B, H, W] -> [B, 1, H, W]
        mask_4d = mask_batch.unsqueeze(1)

        # 创建高斯核
        kernel_size = int(feathering * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = feathering / 3.0

        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                         dtype=torch.float32, device=processing_device)
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # 批量应用高斯模糊
        padding = kernel_size // 2
        blurred_mask = F.conv2d(mask_4d, gaussian_2d, padding=padding)

        # [B, 1, H, W] -> [B, H, W]
        return blurred_mask.squeeze(1)

    def restore_crop_box(self, background_image, croped_image, crop_box,
                         feathering=0, device="GPU", croped_mask=None):
        """
        PyTorch 批处理版本的 restore

        Args:
            background_image: [B, H, W, C] 背景图像批次
            croped_image: [B, H', W', C] 裁剪图像批次
            crop_box: [x1, y1, x2, y2] 裁剪框位置
            feathering: 羽化半径
            device: CPU/GPU
            croped_mask: 可选的 mask

        Returns:
            restored_images: [B, H, W, C]
            restored_masks: [B, H, W]
        """
        # 批次大小
        batch_size_bg = background_image.shape[0]
        batch_size_crop = croped_image.shape[0]
        batch_size = batch_size_bg  # 使用背景图像的批次大小

        H, W, C = background_image.shape[1:]
        H_crop, W_crop = croped_image.shape[1:3]

        # 处理批次不匹配的情况
        if batch_size_crop == 1 and batch_size_bg > 1:
            # croped_image 只有 1 帧，复制到 batch_size 帧
            # 使用 repeat 而不是 expand，确保内存独立
            croped_image = croped_image.repeat(batch_size, 1, 1, 1)
            log(f"{self.NODE_NAME}: ⚠️ Auto-expanding: croped_image has only 1 frame, "
                f"repeating it to match background's {batch_size} frames. "
                f"This is correct for applying a single AI-processed result to a video. "
                f"If you expect per-frame processing, check your upstream nodes (e.g., use ImageBatchToImageList + ImageListToImageBatch).",
                message_type='warning')
        elif batch_size_crop != batch_size_bg:
            # 批次大小不匹配，取最小值并警告
            batch_size = min(batch_size_bg, batch_size_crop)
            background_image = background_image[:batch_size]
            croped_image = croped_image[:batch_size]
            log(f"{self.NODE_NAME}: ⚠️ Batch size mismatch detected: background has {batch_size_bg} frames, "
                f"croped_image has {batch_size_crop} frames. Using first {batch_size} frames from both. "
                f"Check your workflow to ensure correct frame counts.",
                message_type='warning')

        # 选择设备
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        # 移动到设备
        background_image = background_image.to(processing_device)
        croped_image = croped_image.to(processing_device)

        # 解析 crop_box
        x1, y1, x2, y2 = crop_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 裁剪框尺寸
        crop_width = x2 - x1
        crop_height = y2 - y1

        log(f"{self.NODE_NAME}: Batch processing {batch_size} frames "
            f"(crop from {croped_image.shape} to position ({x1},{y1},{x2},{y2}))")

        # === 1. 处理 Mask ===
        if croped_mask is not None:
            # 使用提供的 mask
            if croped_mask.dim() == 2:
                # 单张 mask，扩展到批次
                mask_batch = croped_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # 批次 mask
                mask_batch = croped_mask[:batch_size]
            mask_batch = mask_batch.to(processing_device)
        else:
            # 使用全白 mask（所有像素都粘贴）
            mask_batch = torch.ones(batch_size, H_crop, W_crop,
                                    dtype=torch.float32, device=processing_device)

        # 确保 mask 尺寸匹配
        if mask_batch.shape[1:3] != (H_crop, W_crop):
            # 调整 mask 尺寸以匹配 croped_image
            mask_batch = F.interpolate(
                mask_batch.unsqueeze(1),  # [B, 1, H, W]
                size=(H_crop, W_crop),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H_crop, W_crop]

        # 应用羽化
        if feathering > 0:
            mask_batch = self._create_feathered_mask_batch(mask_batch, feathering, device)

        # === 2. Resize croped_image 和 mask 到 crop_box 尺寸（如果需要）===
        if H_crop != crop_height or W_crop != crop_width:
            # 需要调整裁剪图像尺寸
            # [B, H, W, C] -> [B, C, H, W]
            croped_image_resized = croped_image.permute(0, 3, 1, 2)
            croped_image_resized = F.interpolate(
                croped_image_resized,
                size=(crop_height, crop_width),
                mode='bilinear',
                align_corners=False
            )
            # [B, C, H, W] -> [B, H, W, C]
            croped_image_resized = croped_image_resized.permute(0, 2, 3, 1)

            # 调整 mask
            mask_batch = F.interpolate(
                mask_batch.unsqueeze(1),  # [B, 1, H, W]
                size=(crop_height, crop_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H, W]
        else:
            croped_image_resized = croped_image

        # === 3. 批量粘贴（PyTorch 矢量化操作）===
        # 复制背景
        result_images = background_image.clone()

        # 扩展 mask 到 3 通道: [B, H, W] -> [B, H, W, C]
        mask_3d = mask_batch.unsqueeze(-1).expand(-1, -1, -1, C)

        # 批量混合：result = background * (1 - mask) + cropped * mask
        # 只在 crop_box 区域内操作
        result_images[:, y1:y2, x1:x2, :] = (
            result_images[:, y1:y2, x1:x2, :] * (1 - mask_3d) +
            croped_image_resized * mask_3d
        )

        # === 4. 创建输出 mask ===
        # 全黑背景 mask
        result_masks = torch.zeros(batch_size, H, W, dtype=torch.float32, device=processing_device)
        # 粘贴区域设置为 mask 值
        result_masks[:, y1:y2, x1:x2] = mask_batch

        log(f"{self.NODE_NAME} Processed {batch_size} frames in batch mode.", message_type='finish')

        # 返回到原始设备（通常是 CPU，供后续节点使用）
        result_images = result_images.cpu()
        result_masks = result_masks.cpu()

        return (result_images, result_masks)


NODE_CLASS_MAPPINGS = {
    "SwwanRestoreCropBoxV4": RestoreCropBoxV4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwwanRestoreCropBoxV4": "Restore Crop Box V4 (Fast)"
}
