# SPDX-License-Identifier: GPL-3.0-only
# Adapted from https://github.com/goohai/Goohaitools-comfyui
# Source revision: a84303e6e73a289af59d96eddab9f521ebd00643
# Original algorithms retained; Swwan wrappers are registered separately.
# See licenses/GPL-3.0.txt and THIRD_PARTY_NOTICES.md.

# BlockifyMask adapted from kijai/ComfyUI-KJNodes, revision 6ab7e8130e449ed2c0037589bcf84146ceb7fc9c.

import colorsys
import re

import numpy as np
import torch
from PIL import Image

HIGH_QUALITY_INTERPOLATION = Image.LANCZOS
MASK_INTERPOLATION = Image.BILINEAR


class ImageResizeRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "限制模式": (["长边", "短边", "宽度", "高度", "宽度与高度"], {"default": "长边"}),
                "最小尺寸": ("INT", {"default": 1024, "min": 10, "max": 10240, "step": 1}),
                "最大尺寸": ("INT", {"default": 3000, "min": 10, "max": 10240, "step": 1}),
                "整除数": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "resize_image_range"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "把图像与遮罩尺寸限制到指定范围 - 支持多种限制模式的高质量图像缩放"

    def resize_image_range(self, 图像=None, 遮罩=None, 限制模式="长边", 最小尺寸=1024, 最大尺寸=3000, 整除数=0):
        if 遮罩 is not None and 遮罩.ndim == 2:
            遮罩 = 遮罩.unsqueeze(0)

        # 检查至少有一个输入
        if 图像 is None and 遮罩 is None:
            raise ValueError("错误: 至少需要输入图像或遮罩中的一个")

        # 检查尺寸匹配
        if 图像 is not None and 遮罩 is not None:
            batch_size_img = 图像.shape[0] if 图像.ndim == 4 else 1
            batch_size_mask = 遮罩.shape[0] if 遮罩.ndim == 3 else 1

            if batch_size_img != batch_size_mask:
                raise ValueError(f"错误: 图像和遮罩的批次大小不匹配: 图像={batch_size_img}, 遮罩={batch_size_mask}")

        # 确保最小尺寸和最大尺寸合理
        最小尺寸 = max(1, 最小尺寸)
        最大尺寸 = max(最小尺寸, 最大尺寸)

        # 处理图像
        resized_images = []
        if 图像 is not None:
            for img in 图像:
                resized_img = self.resize_single_image(img, 限制模式, 最小尺寸, 最大尺寸, 整除数, True)
                resized_images.append(resized_img)

            if resized_images:
                resized_images = torch.cat(resized_images, dim=0)
            else:
                resized_images = None

        # 处理遮罩
        resized_masks = []
        if 遮罩 is not None:
            # 检查遮罩是否全黑
            all_black_mask = True
            for mask in 遮罩:
                if not self.is_black_mask(mask):
                    all_black_mask = False
                    break

            # 如果遮罩全黑，则跳过处理，后面会生成全黑遮罩
            if not all_black_mask or 图像 is None:
                for mask in 遮罩:
                    # 为遮罩添加通道维度以匹配图像格式
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(-1)

                    resized_mask = self.resize_single_image(mask, 限制模式, 最小尺寸, 最大尺寸, 整除数, False)
                    resized_masks.append(resized_mask)

                if resized_masks:
                    resized_masks = torch.cat(resized_masks, dim=0)
                    # 移除通道维度，恢复遮罩格式
                    if resized_masks.shape[-1] == 1:
                        resized_masks = resized_masks.squeeze(-1)
                else:
                    resized_masks = None
            else:
                resized_masks = None  # 全黑遮罩，后面会生成
        else:
            resized_masks = None  # 没有遮罩输入，后面会生成全黑遮罩

        # 生成全黑遮罩的逻辑
        if 图像 is not None and (遮罩 is None or (遮罩 is not None and self.is_black_mask(遮罩))):
            # 获取输出图像的尺寸
            if resized_images is not None:
                batch_size = resized_images.shape[0]
                height = resized_images.shape[1]
                width = resized_images.shape[2]

                # 创建全黑遮罩
                resized_masks = torch.zeros((batch_size, height, width), dtype=torch.float32)

        # 如果只有遮罩输入，图像输出为None
        if 图像 is None and resized_masks is not None:
            resized_images = None

        return (resized_images, resized_masks)

    def is_black_mask(self, mask_tensor):
        """
        检查遮罩是否全黑
        """
        if mask_tensor is None:
            return True

        # 处理批次维度
        if mask_tensor.ndim == 3:  # [B, H, W]
            for i in range(mask_tensor.shape[0]):
                if torch.any(mask_tensor[i] > 0.01):  # 容忍很小的误差
                    return False
            return True
        elif mask_tensor.ndim == 2:  # [H, W]
            return not torch.any(mask_tensor > 0.01)
        else:
            # 其他维度格式，默认不全黑
            return False

    def resize_single_image(self, img_tensor, 限制模式, 最小尺寸, 最大尺寸, 整除数, is_image=True):
        """
        处理单个图像或遮罩
        """
        # 转换为PIL图像
        if is_image:
            # 图像: [H, W, C] -> PIL Image
            img_pil = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
        else:
            # 遮罩: [H, W, 1] -> PIL Image (L mode)
            if img_tensor.shape[-1] == 1:
                mask_array = (img_tensor.cpu().numpy().squeeze(-1) * 255).astype(np.uint8)
            else:
                mask_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(mask_array, mode='L')

        original_width, original_height = img_pil.size
        原始比例 = original_width / original_height

        # 根据限制模式计算目标尺寸
        目标宽度, 目标高度 = self.calculate_target_size(
            original_width, original_height, 限制模式, 最小尺寸, 最大尺寸, 原始比例
        )

        # 处理整除数 - 使用覆盖裁剪模式
        if 整除数 > 0:
            目标宽度, 目标高度 = self.apply_divisor_with_crop(
                目标宽度, 目标高度, 整除数, 原始比例
            )

        # 确保最小尺寸
        目标宽度 = max(1, 目标宽度)
        目标高度 = max(1, 目标高度)

        # 选择插值方法
        interpolation = HIGH_QUALITY_INTERPOLATION if is_image else MASK_INTERPOLATION

        # 先进行等比例缩放
        scaled_pil = self.resize_with_aspect_ratio(img_pil, 目标宽度, 目标高度, interpolation)

        # 如果整除数>0，进行居中裁剪到精确的倍数
        if 整除数 > 0:
            scaled_pil = self.center_crop_to_divisor(scaled_pil, 整除数)

        # 转换回tensor
        if is_image:
            resized_array = np.array(scaled_pil).astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_array)
        else:
            resized_array = np.array(scaled_pil).astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_array)
            if resized_tensor.ndim == 2:
                resized_tensor = resized_tensor.unsqueeze(-1)

        return resized_tensor.unsqueeze(0)

    def apply_divisor_with_crop(self, 宽度, 高度, 整除数, 原始比例):
        """
        使用覆盖裁剪模式处理整除数：一个方向铺满，另一个方向居中裁剪
        """
        # 计算两个可能的方案
        方案1_宽度 = (宽度 // 整除数) * 整除数
        方案1_高度 = int(方案1_宽度 / 原始比例)
        方案1_高度 = (方案1_高度 // 整除数) * 整除数

        方案2_高度 = (高度 // 整除数) * 整除数
        方案2_宽度 = int(方案2_高度 * 原始比例)
        方案2_宽度 = (方案2_宽度 // 整除数) * 整除数

        # 选择更接近原始尺寸的方案
        方案1_面积 = 方案1_宽度 * 方案1_高度
        方案2_面积 = 方案2_宽度 * 方案2_高度
        目标面积 = 宽度 * 高度

        if abs(方案1_面积 - 目标面积) <= abs(方案2_面积 - 目标面积):
            return 方案1_宽度, 方案1_高度
        else:
            return 方案2_宽度, 方案2_高度

    def resize_with_aspect_ratio(self, img_pil, 目标宽度, 目标高度, interpolation):
        """
        保持宽高比进行缩放
        """
        当前宽度, 当前高度 = img_pil.size
        当前比例 = 当前宽度 / 当前高度
        目标比例 = 目标宽度 / 目标高度

        if 当前比例 > 目标比例:
            # 宽度较大，先缩放高度，然后裁剪宽度
            缩放高度 = 目标高度
            缩放宽度 = int(缩放高度 * 当前比例)
        else:
            # 高度较大，先缩放宽度，然后裁剪高度
            缩放宽度 = 目标宽度
            缩放高度 = int(缩放宽度 / 当前比例)

        # 等比例缩放
        scaled_img = img_pil.resize((缩放宽度, 缩放高度), interpolation)
        return scaled_img

    def center_crop_to_divisor(self, img_pil, 整除数):
        """
        居中裁剪到整除数的倍数
        """
        当前宽度, 当前高度 = img_pil.size

        # 计算裁剪后的尺寸（整除数的倍数）
        裁剪宽度 = (当前宽度 // 整除数) * 整除数
        裁剪高度 = (当前高度 // 整除数) * 整除数

        # 确保裁剪尺寸有效
        裁剪宽度 = max(整除数, 裁剪宽度)
        裁剪高度 = max(整除数, 裁剪高度)

        # 计算裁剪区域（居中）
        左边 = (当前宽度 - 裁剪宽度) // 2
        上边 = (当前高度 - 裁剪高度) // 2
        右边 = 左边 + 裁剪宽度
        下边 = 上边 + 裁剪高度

        # 执行裁剪
        cropped_img = img_pil.crop((左边, 上边, 右边, 下边))
        return cropped_img

    def calculate_target_size(self, 原始宽度, 原始高度, 限制模式, 最小尺寸, 最大尺寸, 原始比例):
        """
        根据限制模式计算目标尺寸
        """
        if 限制模式 == "长边":
            长边 = max(原始宽度, 原始高度)
            if 长边 < 最小尺寸:
                缩放比例 = 最小尺寸 / 长边
            elif 长边 > 最大尺寸:
                缩放比例 = 最大尺寸 / 长边
            else:
                缩放比例 = 1.0

            目标宽度 = int(round(原始宽度 * 缩放比例))
            目标高度 = int(round(原始高度 * 缩放比例))

        elif 限制模式 == "短边":
            短边 = min(原始宽度, 原始高度)
            if 短边 < 最小尺寸:
                缩放比例 = 最小尺寸 / 短边
            elif 短边 > 最大尺寸:
                缩放比例 = 最大尺寸 / 短边
            else:
                缩放比例 = 1.0

            目标宽度 = int(round(原始宽度 * 缩放比例))
            目标高度 = int(round(原始高度 * 缩放比例))

        elif 限制模式 == "宽度":
            if 原始宽度 < 最小尺寸:
                缩放比例 = 最小尺寸 / 原始宽度
            elif 原始宽度 > 最大尺寸:
                缩放比例 = 最大尺寸 / 原始宽度
            else:
                缩放比例 = 1.0

            目标宽度 = int(round(原始宽度 * 缩放比例))
            目标高度 = int(round(原始高度 * 缩放比例))

        elif 限制模式 == "高度":
            if 原始高度 < 最小尺寸:
                缩放比例 = 最小尺寸 / 原始高度
            elif 原始高度 > 最大尺寸:
                缩放比例 = 最大尺寸 / 原始高度
            else:
                缩放比例 = 1.0

            目标宽度 = int(round(原始宽度 * 缩放比例))
            目标高度 = int(round(原始高度 * 缩放比例))

        elif 限制模式 == "宽度与高度":
            # 这个模式下需要特殊处理
            较小边 = min(原始宽度, 原始高度)
            较大边 = max(原始宽度, 原始高度)

            if 较小边 < 最小尺寸:
                # 较小边缩放到最小尺寸
                缩放比例 = 最小尺寸 / 较小边
                目标宽度 = int(round(原始宽度 * 缩放比例))
                目标高度 = int(round(原始高度 * 缩放比例))

                # 检查较大边是否超过最大尺寸
                if max(目标宽度, 目标高度) > 最大尺寸:
                    # 需要裁剪
                    pass
            elif 较大边 > 最大尺寸:
                # 尝试缩放到最大尺寸
                缩放比例 = 最大尺寸 / 较大边
                目标宽度 = int(round(原始宽度 * 缩放比例))
                目标高度 = int(round(原始高度 * 缩放比例))

                # 检查较小边是否小于最小尺寸
                if min(目标宽度, 目标高度) < 最小尺寸:
                    # 重新缩放，以较小边缩放到最小尺寸
                    缩放比例 = 最小尺寸 / 较小边
                    目标宽度 = int(round(原始宽度 * 缩放比例))
                    目标高度 = int(round(原始高度 * 缩放比例))
            else:
                # 尺寸已经在范围内
                目标宽度 = 原始宽度
                目标高度 = 原始高度
        else:
            # 默认不缩放
            目标宽度 = 原始宽度
            目标高度 = 原始高度

        return 目标宽度, 目标高度


class ColorConverter:
    """颜色转换节点 - 孤海"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "色值": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
                "转换后": (["#HEX", "HEX", "RGB", "HSL"], {
                    "default": "#HEX"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "COLORCODE")
    RETURN_NAMES = ("字符串", "颜色控件")
    FUNCTION = "convert_color"
    CATEGORY = "Swwan/image"
    OUTPUT_NODE = False

    def normalize_symbols(self, text):
        """将全角符号转换为半角符号"""
        text = text.replace('，', ',')
        text = text.replace('（', '(')
        text = text.replace('）', ')')
        text = text.replace('　', ' ')
        return text

    def parse_color(self, color_str):
        """解析多种格式的颜色值"""
        color_str = str(color_str).strip().lower()
        color_str = self.normalize_symbols(color_str)
        color_str = re.sub(r'\s+', '', color_str)

        hex_match = re.match(r'^#?([0-9a-f]{3}|[0-9a-f]{6})$', color_str)
        if hex_match:
            hex_code = hex_match.group(1)
            if len(hex_code) == 3:
                hex_code = ''.join([c*2 for c in hex_code])
            return self.hex_to_rgb(hex_code)

        rgb_pattern = r'^[\(（]?\s*(\d{1,3})\s*[，,]\s*(\d{1,3})\s*[，,]\s*(\d{1,3})\s*[\)）]?$'
        rgb_match = re.match(rgb_pattern, color_str)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r/255, g/255, b/255)

        hsl_pattern = r'^[\(（]?\s*(\d{1,3})\s*[，,]\s*(\d{1,3})%\s*[，,]\s*(\d{1,3})%\s*[\)）]?$'
        hsl_match = re.match(hsl_pattern, color_str)
        if hsl_match:
            h, s, l_val = map(float, hsl_match.groups())
            h = h / 360.0
            s = s / 100.0
            l_val = l_val / 100.0
            return self.hsl_to_rgb_normalized(h, s, l_val)

        try:
            if re.match(r'^[01]\.\d+\s*[，,]\s*[01]\.\d+\s*[，,]\s*[01]\.\d+$', color_str):
                parts = re.split(r'[，,]\s*', color_str)
                if len(parts) == 3:
                    r, g, b = map(float, parts)
                    if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
                        return (r, g, b)
        except:
            pass

        return (1.0, 1.0, 1.0)

    def hex_to_rgb(self, hex_code):
        """十六进制转RGB(0-1范围)"""
        hex_code = hex_code.lstrip('#')
        if len(hex_code) == 3:
            hex_code = ''.join([c*2 for c in hex_code])
        r = int(hex_code[0:2], 16) / 255.0
        g = int(hex_code[2:4], 16) / 255.0
        b = int(hex_code[4:6], 16) / 255.0
        return (r, g, b)

    def rgb_to_hex(self, r, g, b, with_hash=True):
        """RGB(0-1范围)转十六进制"""
        r_int = int(min(max(r * 255, 0), 255))
        g_int = int(min(max(g * 255, 0), 255))
        b_int = int(min(max(b * 255, 0), 255))
        hex_code = f"{r_int:02x}{g_int:02x}{b_int:02x}"
        return f"#{hex_code}" if with_hash else hex_code

    def rgb_to_hsl_normalized(self, r, g, b):
        """RGB(0-1范围)转HSL(0-360, 0-100%, 0-100%)"""
        h, l_val, s = colorsys.rgb_to_hls(r, g, b)
        h = (h * 360) % 360
        s = s * 100
        l_val = l_val * 100
        return h, s, l_val

    def hsl_to_rgb_normalized(self, h, s, l_val):
        """HSL(0-1范围)转RGB(0-1范围)"""
        r, g, b = colorsys.hls_to_rgb(h, l_val, s)
        return (r, g, b)

    def convert_color(self, 色值, 转换后):
        """转换颜色格式"""
        rgb_normalized = self.parse_color(色值)

        if 转换后 == "#HEX":
            result_str = self.rgb_to_hex(*rgb_normalized, with_hash=True)
        elif 转换后 == "HEX":
            result_str = self.rgb_to_hex(*rgb_normalized, with_hash=False)
        elif 转换后 == "RGB":
            r = int(rgb_normalized[0] * 255)
            g = int(rgb_normalized[1] * 255)
            b = int(rgb_normalized[2] * 255)
            result_str = f"{r},{g},{b}"
        elif 转换后 == "HSL":
            h, s, l_val = self.rgb_to_hsl_normalized(*rgb_normalized)
            result_str = f"{int(round(h))},{int(round(s))}%,{int(round(l_val))}%"
        else:
            result_str = "#ffffff"

        color_control = self.rgb_to_hex(*rgb_normalized, with_hash=True)

        return (result_str, color_control)


class BlockifyMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "block_size": ("INT", {"default": 32, "min": 8, "max": 512, "step": 1, "tooltip": "Size of blocks in pixels (smaller = smaller blocks)"}),
                },
                "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Device to use for processing"}),
                }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "Swwan/masking"
    DESCRIPTION = "Creates a block mask by dividing the bounding box of each mask into blocks of the specified size and filling in blocks that contain any part of the original mask."

    def process(self, masks, block_size, device="cpu"):
        if device == "gpu":
            from comfy import model_management
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        masks = masks.to(processing_device)
        batch_size, height, width = masks.shape

        result_masks = torch.zeros_like(masks)

        for i in range(batch_size):
            mask = masks[i]

            # Find bounding box efficiently
            mask_bool = mask > 0
            if not mask_bool.any():
                continue

            y_indices = torch.nonzero(mask_bool.any(dim=1), as_tuple=True)[0]
            x_indices = torch.nonzero(mask_bool.any(dim=0), as_tuple=True)[0]

            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]

            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1

            # Calculate block grid
            w_divisions = max(1, bbox_width // block_size)
            h_divisions = max(1, bbox_height // block_size)

            w_slice = bbox_width // w_divisions
            h_slice = bbox_height // h_divisions

            # Create coordinate grids only for bbox region
            y_coords = torch.arange(y_min, y_max + 1, device=processing_device).view(-1, 1)
            x_coords = torch.arange(x_min, x_max + 1, device=processing_device).view(1, -1)

            # Calculate block indices for bbox region
            w_block_indices = (x_coords - x_min) // w_slice
            h_block_indices = (y_coords - y_min) // h_slice

            # Clamp to valid range
            w_block_indices = w_block_indices.clamp(0, w_divisions - 1)
            h_block_indices = h_block_indices.clamp(0, h_divisions - 1)

            # Create unique block IDs by combining h and w indices
            block_ids = h_block_indices * w_divisions + w_block_indices

            # Get mask region within bbox
            mask_region = mask[y_min:y_max+1, x_min:x_max+1]

            # Find which blocks have content using scatter_add
            max_blocks = h_divisions * w_divisions
            block_content = torch.zeros(max_blocks, device=processing_device)
            block_content.scatter_add_(0, block_ids.flatten(), mask_region.flatten())

            # Create result for blocks that have content
            has_content = block_content > 0
            block_mask = has_content[block_ids]

            # Fill the result
            result_masks[i, y_min:y_max+1, x_min:x_max+1] = block_mask.float()

        return (result_masks.clamp(0, 1),)


class ImagesToRGB:
    """PIL RGB conversion matching WAS Images to RGB without importing WAS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "Swwan/image"

    def convert(self, images):
        if images.numel() == 0:
            raise ValueError("RGB conversion requires at least one image")
        # Match WAS's dynamic-range folding before its 8-bit PIL conversion.
        scale = 1.0
        if float(images.amax()) > 1.001 or float(images.amin()) < -0.001:
            scale = max(1.0, float(images.amax()))
        folded = (images / scale).clamp(0, 1) if scale > 1.001 else images
        if folded.ndim >= 4:
            planes = list(folded)
        elif folded.ndim == 3 and folded.shape[-1] not in (1, 3, 4):
            planes = list(folded)
        else:
            planes = [folded]
        result = []
        for plane in planes:
            array = np.clip(255.0 * plane.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            rgb = Image.fromarray(array).convert("RGB")
            result.append(torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0))
        output = torch.stack(result)
        if scale > 1.001:
            output *= scale
        dtype = images.dtype if images.is_floating_point() else torch.float32
        return (output.to(device=images.device, dtype=dtype),)
