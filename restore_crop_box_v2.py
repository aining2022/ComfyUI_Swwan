import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management
from .layerstyle_utils import log, tensor2pil, pil2tensor, image2mask


class RestoreCropBoxV2:
    """
    RestoreCropBox V2 - 增强版裁剪还原节点

    相比 V1 新增功能：
    - feathering: 边缘羽化，解决粘贴时的色差/硬边问题
    - device: 支持 CPU/GPU 选择
    """

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox V2'

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
    CATEGORY = 'RunningHub/LayerUtility'
    DESCRIPTION = """RestoreCropBox V2 - 将裁剪的图像还原到原始背景

新增参数:
- feathering: 边缘羽化半径，让边缘柔和过渡，解决色差问题
- device: 选择 CPU 或 GPU 进行处理"""

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

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []

        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in croped_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))

        if croped_mask is not None:
            if croped_mask.dim() == 2:
                croped_mask = torch.unsqueeze(croped_mask, 0)
            l_masks = []
            for m in croped_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            croped_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(croped_image).convert('RGB')

            # 应用羽化
            _mask = self._apply_feathering(_mask, feathering, device)

            ret_mask = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
            ret_mask.paste(_mask, box=tuple(crop_box))
            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(ret_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: RestoreCropBox V2": RestoreCropBoxV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RestoreCropBox V2": "LayerUtility: RestoreCropBox V2"
}
