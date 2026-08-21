import torch
import torch.nn.functional as F

from comfy import model_management
from .layerstyle_utils import log


class ColorShiftFix:
    """
    Color Shift Fix (Swwan) - 视频 inpaint 色差修复

    原理：
    LTX 等视频模型重生成后会产生全局色偏（对比度压缩 + 提亮，偏暖）。
    本节点以「遮罩外未被 inpaint 的区域」为可信参考，对每帧的 R/G/B 通道
    拟合仿射变换 corrected = gain * x + offset（迭代修剪最小二乘，自动剔除
    内容本身变化的像素），再对参数序列做时序滑动平均，避免逐帧闪烁。

    接线（ltx 去水印工作流）：
    - images:           VAE Decode (Tiled) 输出
    - reference_images: CropByMask V4 的 croped_image（同裁剪区的原始帧）
    - mask:             ImageToMask（裁剪后的遮罩，5200）
    - 输出接 Restore Crop Box V4 (Fast) 的 croped_image
    """

    def __init__(self):
        self.NODE_NAME = 'ColorShiftFix'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth_window": ("INT", {"default": 9, "min": 1, "max": 99, "step": 2}),
                "trim_percentile": ("FLOAT", {"default": 70.0, "min": 50.0, "max": 95.0, "step": 5.0}),
                "device": (["CPU", "GPU"], {"default": "GPU"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'fix_color_shift'
    CATEGORY = 'Swwan/Image'
    DESCRIPTION = """Color Shift Fix - 视频 inpaint 色差修复

原理:
- 以遮罩外未修改区域为参考，逐帧拟合每通道仿射变换 (gain/offset)
- 迭代修剪最小二乘，自动剔除内容变化像素（被抹除的水印/底栏）
- 参数时序滑动平均，防止视频颜色闪烁

推荐配置:
- smooth_window: 9（约 0.4s @24fps；画面切换快的视频可调小）
- trim_percentile: 70
- strength: 1.0

注意:
- reference_images 必须与 images 内容对齐（同一裁剪区的原始帧）
- mask 缺省时用整帧做参考拟合"""

    def _fit_frame(self, out_img, ref_img, unmasked, trim_percentile):
        """
        单帧拟合每通道仿射变换

        Args:
            out_img: [H, W, C] 待校正帧
            ref_img: [H, W, C] 参考帧
            unmasked: [H, W] bool，True = 可信参考区域
            trim_percentile: 残差修剪百分位

        Returns:
            coefs: [C, 2] (gain, offset)
        """
        C = out_img.shape[-1]
        coefs = torch.zeros(C, 2, dtype=torch.float32, device=out_img.device)
        min_pixels = 256

        # 遮罩外像素太少时退化为整帧拟合
        if unmasked.sum() < min_pixels:
            unmasked = torch.ones_like(unmasked)

        for c in range(C):
            x = out_img[..., c][unmasked]
            y = ref_img[..., c][unmasked]

            keep = torch.ones_like(x, dtype=torch.bool)
            gain, offset = torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device)
            for _ in range(3):
                if keep.sum() < min_pixels:
                    break
                X = torch.stack([x[keep], torch.ones_like(x[keep])], dim=1)
                sol = torch.linalg.lstsq(X, y[keep].unsqueeze(-1)).solution
                gain, offset = sol[0, 0], sol[1, 0]
                resid = (x * gain + offset - y).abs()
                thr = torch.quantile(resid[keep].float(), trim_percentile / 100.0)
                thr = torch.clamp(thr, min=8.0 / 255.0)
                keep = resid <= thr

            coefs[c, 0] = gain
            coefs[c, 1] = offset

        return coefs

    def fix_color_shift(self, images, reference_images, strength=1.0,
                        smooth_window=9, trim_percentile=70.0, device="GPU", mask=None):
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        B, H, W, C = images.shape
        out = images.to(processing_device).float()
        ref = reference_images.to(processing_device).float()

        # 批次对齐：参考帧不足时循环取模
        if ref.shape[0] != B:
            idx = torch.arange(B, device=processing_device) % ref.shape[0]
            ref = ref[idx]

        # 尺寸对齐：参考帧尺寸不同时双线性缩放
        if ref.shape[1] != H or ref.shape[2] != W:
            ref = F.interpolate(ref.permute(0, 3, 1, 2), size=(H, W),
                                mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # mask 对齐：[B, H, W]，缺省视为全遮罩（退化为整帧参考）
        if mask is not None:
            m = mask.to(processing_device).float()
            if m.dim() == 4:
                m = m.squeeze(-1)
            if m.shape[0] != B:
                idx = torch.arange(B, device=processing_device) % m.shape[0]
                m = m[idx]
            if m.shape[1] != H or m.shape[2] != W:
                m = F.interpolate(m.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            unmasked_batch = m < 0.5
        else:
            unmasked_batch = torch.zeros(B, H, W, dtype=torch.bool, device=processing_device)

        # === 1. 逐帧拟合 ===
        all_coefs = torch.zeros(B, C, 2, dtype=torch.float32, device=processing_device)
        for i in range(B):
            all_coefs[i] = self._fit_frame(out[i], ref[i], unmasked_batch[i], trim_percentile)

        # === 2. 参数时序平滑（滑窗均值，边缘复制填充） ===
        if smooth_window > 1 and B > 1:
            win = min(smooth_window, B if B % 2 == 1 else B - 1)
            if win >= 3:
                pad = win // 2
                # [B, C, 2] -> [1, C*2, B]
                coefs_t = all_coefs.reshape(B, -1).permute(1, 0).unsqueeze(0)
                coefs_t = F.pad(coefs_t, (pad, pad), mode='replicate')
                coefs_t = F.avg_pool1d(coefs_t, kernel_size=win, stride=1)
                all_coefs = coefs_t.squeeze(0).permute(1, 0).reshape(B, C, 2)

        # === 3. 应用校正 + 强度混合 ===
        gain = all_coefs[:, :, 0].view(B, 1, 1, C)
        offset = all_coefs[:, :, 1].view(B, 1, 1, C)
        corrected = torch.clamp(out * gain + offset, 0.0, 1.0)
        result = out * (1.0 - strength) + corrected * strength

        log(f"{self.NODE_NAME} Processed {B} frames. "
            f"gain range: [{all_coefs[:, :, 0].min():.3f}, {all_coefs[:, :, 0].max():.3f}], "
            f"offset range: [{all_coefs[:, :, 1].min()*255:.1f}, {all_coefs[:, :, 1].max()*255:.1f}]/255",
            message_type='finish')

        return (result.cpu(),)


NODE_CLASS_MAPPINGS = {
    "SwwanColorShiftFix": ColorShiftFix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwwanColorShiftFix": "Color Shift Fix (Swwan)"
}
