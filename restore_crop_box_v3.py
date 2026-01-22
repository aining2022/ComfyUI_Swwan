import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management
from .layerstyle_utils import log, tensor2pil, pil2tensor, image2mask


class RestoreCropBoxV3:
    """
    RestoreCropBox V3 - 批处理优化版裁剪还原节点

    相比 V2 改进：
    - 简化批处理逻辑，直接支持视频批次输入
    - 自动匹配 background_image 和 croped_image 的批次大小
    - crop_box 在所有帧之间共享（视频处理的标准模式）
    """

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox V3'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_box": ("BOX",),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'Swwan/Image'
    DESCRIPTION = """RestoreCropBox V3 - 批处理裁剪还原节点

批处理改进:
- 自动处理视频批次，无需 ImageList 转换
- background_image 和 croped_image 必须批次大小相同
- 使用同一个 crop_box 应用到所有帧

推荐参数:
- feathering: 低分辨率 15-30, 中分辨率 30-60, 高分辨率 60-120
- device: GPU 处理速度更快"""

    def _apply_feathering(self, mask_pil, feathering, device):
        """
        对 mask 应用羽化效果，实现边缘平滑过渡

        Args:
            mask_pil: PIL Image (L mode)
            feathering: 羽化半径
            device: "CPU" 或 "GPU"

        Returns:
            羽化后的 PIL Image (L mode)
        """
        if feathering <= 0:
            return mask_pil

        # PIL mask -> numpy -> tensor
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)

        # 选择设备
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        mask_tensor = mask_tensor.to(processing_device)
        mask_4d = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # 高斯核参数
        kernel_size = int(feathering * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = feathering / 3.0

        # 创建高斯核
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                         dtype=torch.float32, device=processing_device)
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        # 应用高斯模糊
        padding = kernel_size // 2
        blurred_mask = F.conv2d(mask_4d, gaussian_2d, padding=padding)
        blurred_mask = blurred_mask.squeeze(0).squeeze(0)  # [H, W]

        # 转回 PIL
        result = (blurred_mask.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(result, mode='L')

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         feathering=0, device="CPU", croped_mask=None):

        # 批次大小验证
        batch_size_bg = background_image.shape[0]
        batch_size_crop = croped_image.shape[0]

        if batch_size_bg != batch_size_crop:
            raise ValueError(
                f"Batch size mismatch: background_image has {batch_size_bg} frames, "
                f"but croped_image has {batch_size_crop} frames. They must match."
            )

        batch_size = batch_size_bg
        log(f"{self.NODE_NAME}: Processing {batch_size} frames in batch mode")

        ret_images = []
        ret_masks = []

        # 处理每一帧
        for i in range(batch_size):
            # 获取当前帧
            bg_frame = background_image[i]
            crop_frame = croped_image[i]

            # 转换为 PIL
            _canvas = tensor2pil(torch.unsqueeze(bg_frame, 0)).convert('RGB')
            _layer = tensor2pil(torch.unsqueeze(crop_frame, 0)).convert('RGB')

            # 处理 mask
            if croped_mask is not None:
                # 使用提供的 mask
                if croped_mask.dim() == 2:
                    mask_frame = croped_mask
                else:
                    mask_frame = croped_mask[i] if i < croped_mask.shape[0] else croped_mask[-1]

                if invert_mask:
                    mask_frame = 1 - mask_frame

                _mask = tensor2pil(torch.unsqueeze(mask_frame, 0)).convert('L')
            else:
                # 从裁剪图像提取 alpha 通道或使用白色 mask
                layer_pil = tensor2pil(torch.unsqueeze(crop_frame, 0))
                if layer_pil.mode == 'RGBA':
                    _mask = layer_pil.split()[-1]
                else:
                    _mask = Image.new('L', size=_layer.size, color='white')

            # 应用羽化
            _mask = self._apply_feathering(_mask, feathering, device)

            # 粘贴裁剪图像到背景
            ret_mask = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
            ret_mask.paste(_mask, box=tuple(crop_box))

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(ret_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "SwwanRestoreCropBoxV3": RestoreCropBoxV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwwanRestoreCropBoxV3": "Restore Crop Box V3 (Batch)"
}
