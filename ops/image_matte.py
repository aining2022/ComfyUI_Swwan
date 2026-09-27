# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/保留遮罩区域移除图像背景.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np
import torch
from PIL import Image, ImageFilter

class MaskMatteAlgorithm:
    """
    孤海-保留遮罩区域移除图像背景
    """

    def remove_background(self, 图像, 遮罩, 遮罩填充漏洞, 遮罩裁剪, 裁剪系数, 图像描边, 描边颜色):
        batch_size = 图像.shape[0]
        rgba_results = []
        rgb_results = []
        mask_results = []
        for i in range(batch_size):
            image_tensor = 图像[i]
            mask_tensor = 遮罩[0] if i >= 遮罩.shape[0] else 遮罩[i]
            rgba_tensor, rgb_tensor, mask_tensor = self.process_single_image(image_tensor, mask_tensor, 遮罩填充漏洞, 遮罩裁剪, 裁剪系数, 图像描边, 描边颜色)
            rgba_results.append(rgba_tensor)
            rgb_results.append(rgb_tensor)
            mask_results.append(mask_tensor)
        rgba_batch = torch.cat(rgba_results, dim=0)
        rgb_batch = torch.cat(rgb_results, dim=0)
        mask_batch = torch.cat(mask_results, dim=0)
        return (rgba_batch, rgb_batch, mask_batch)

    def process_single_image(self, image_tensor, mask_tensor, fill_holes, crop_mask, crop_factor, stroke_width, stroke_color):
        image_pil = self.tensor2pil(image_tensor)
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor.squeeze(0)
        mask_array = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_array, mode='L')
        image_rgba = image_pil.convert('RGBA')
        if fill_holes:
            mask_array = np.array(mask_pil)
            mask_array = self.fill_mask_holes(mask_array)
            mask_pil = Image.fromarray(mask_array, mode='L')
        if crop_mask:
            stroke_expansion = max(0, stroke_width) if stroke_width > 0 else 0
            image_rgba, mask_pil = self.crop_with_expansion(image_rgba, mask_pil, crop_factor, stroke_expansion)
        elif mask_pil.size != image_rgba.size:
            mask_pil = mask_pil.resize(image_rgba.size, Image.NEAREST)
        result_rgba = self.apply_mask_to_image(image_rgba, mask_pil)
        if stroke_width > 0:
            result_rgba = self.add_image_stroke(result_rgba, mask_pil, stroke_width, stroke_color)
        rgb_image = self.rgba_to_rgb(result_rgba, stroke_color)
        output_mask = self.pil_to_mask(mask_pil)
        rgba_tensor = self.pil2tensor(result_rgba)
        rgb_tensor = self.pil2tensor(rgb_image)
        mask_tensor = output_mask.unsqueeze(0)
        return (rgba_tensor, rgb_tensor, mask_tensor)

    def fill_mask_holes(self, mask_array):
        """填充遮罩内部的孔洞"""
        try:
            from scipy import ndimage
            filled_mask = ndimage.binary_fill_holes(mask_array > 128)
            return (filled_mask * 255).astype(np.uint8)
        except ImportError:
            print('警告: scipy不可用，使用简单孔洞填充方法')
            return self.simple_fill_holes(mask_array)

    def simple_fill_holes(self, mask_array):
        """简单的孔洞填充方法（不使用scipy）"""
        filled = mask_array.copy()
        h, w = filled.shape

        def flood_fill(x, y, target, replacement):
            stack = [(x, y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= w or y < 0 or (y >= h):
                    continue
                if filled[y, x] != target:
                    continue
                filled[y, x] = replacement
                stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        for x in range(w):
            if filled[0, x] < 128:
                flood_fill(x, 0, filled[0, x], 255)
            if filled[h - 1, x] < 128:
                flood_fill(x, h - 1, filled[h - 1, x], 255)
        for y in range(h):
            if filled[y, 0] < 128:
                flood_fill(0, y, filled[y, 0], 255)
            if filled[y, w - 1] < 128:
                flood_fill(w - 1, y, filled[y, w - 1], 255)
        filled = 255 - filled
        return filled

    def crop_with_expansion(self, image, mask, expansion_factor, stroke_expansion=0):
        """根据遮罩区域裁剪图像并进行扩展，确保描边可见"""
        mask_array = np.array(mask)
        coords = np.column_stack(np.where(mask_array > 128))
        if len(coords) == 0:
            return (image, mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        expand_w = int(width * (expansion_factor - 1.0) / 2) + stroke_expansion
        expand_h = int(height * (expansion_factor - 1.0) / 2) + stroke_expansion
        new_x_min = max(0, x_min - expand_w)
        new_y_min = max(0, y_min - expand_h)
        new_x_max = min(image.width, x_max + expand_w)
        new_y_max = min(image.height, y_max + expand_h)
        left_expand = max(0, expand_w - x_min)
        top_expand = max(0, expand_h - y_min)
        right_expand = max(0, x_max + expand_w - image.width)
        bottom_expand = max(0, y_max + expand_h - image.height)
        if left_expand > 0 or top_expand > 0 or right_expand > 0 or (bottom_expand > 0):
            canvas_width = new_x_max - new_x_min + left_expand + right_expand
            canvas_height = new_y_max - new_y_min + top_expand + bottom_expand
            new_image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            new_mask = Image.new('L', (canvas_width, canvas_height), 0)
            paste_x = left_expand
            paste_y = top_expand
            crop_x_min = max(0, new_x_min)
            crop_y_min = max(0, new_y_min)
            crop_x_max = min(image.width, new_x_max)
            crop_y_max = min(image.height, new_y_max)
            if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                cropped_image = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                cropped_mask = mask.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                new_image.paste(cropped_image, (paste_x, paste_y))
                new_mask.paste(cropped_mask, (paste_x, paste_y))
            return (new_image, new_mask)
        else:
            crop_box = (new_x_min, new_y_min, new_x_max, new_y_max)
            return (image.crop(crop_box), mask.crop(crop_box))

    def apply_mask_to_image(self, image, mask):
        """应用遮罩到图像，挖空背景"""
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        alpha = np.array(mask)
        alpha = np.where(alpha > 128, 255, 0).astype(np.uint8)
        rgb = np.array(image.convert('RGB'))
        rgba = np.dstack((rgb, alpha))
        return Image.fromarray(rgba, 'RGBA')

    def add_image_stroke(self, image, mask, stroke_width, stroke_color):
        """添加图像描边，确保描边宽度一致且完全可见"""
        if stroke_width <= 0:
            return image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        if isinstance(stroke_color, str):
            stroke_rgb = self.hex_to_rgb(stroke_color)
        else:
            stroke_rgb = self.parse_color_object(stroke_color)
        stroke_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        temp_mask = mask.copy()
        for i in range(stroke_width):
            temp_mask = temp_mask.filter(ImageFilter.MaxFilter(3))
        stroke_only = Image.new('L', mask.size, 0)
        stroke_only_array = np.array(temp_mask) - np.array(mask)
        stroke_only_array = np.clip(stroke_only_array, 0, 255).astype(np.uint8)
        stroke_only = Image.fromarray(stroke_only_array, 'L')
        stroke_data = np.array(stroke_image)
        stroke_only_array = np.array(stroke_only)
        stroke_positions = stroke_only_array > 0
        stroke_data[stroke_positions, 0] = stroke_rgb[0]
        stroke_data[stroke_positions, 1] = stroke_rgb[1]
        stroke_data[stroke_positions, 2] = stroke_rgb[2]
        stroke_data[stroke_positions, 3] = 255
        stroke_image = Image.fromarray(stroke_data, 'RGBA')
        result = Image.alpha_composite(stroke_image, image)
        return result

    def rgba_to_rgb(self, rgba_image, stroke_color):
        """将RGBA图像转换为RGB图像，透明背景填充为指定颜色"""
        if isinstance(stroke_color, str):
            bg_rgb = self.hex_to_rgb(stroke_color)
        else:
            bg_rgb = self.parse_color_object(stroke_color)
        bg_image = Image.new('RGB', rgba_image.size, bg_rgb)
        rgba_array = np.array(rgba_image)
        rgb_array = np.array(bg_image)
        alpha = rgba_array[:, :, 3:] / 255.0
        rgb_result = (rgba_array[:, :, :3] * alpha + rgb_array * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(rgb_result, 'RGB')

    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB元组"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return tuple((int(hex_color[i:i + 2], 16) for i in (0, 2, 4)))

    def parse_color_object(self, color_obj):
        """解析ComfyUI颜色对象"""
        if isinstance(color_obj, dict):
            if 'r' in color_obj and 'g' in color_obj and ('b' in color_obj):
                return (color_obj['r'], color_obj['g'], color_obj['b'])
            elif 'hex' in color_obj:
                return self.hex_to_rgb(color_obj['hex'])
        if isinstance(color_obj, (list, tuple)) and len(color_obj) >= 3:
            return tuple((int(c) for c in color_obj[:3]))
        return (255, 255, 255)

    def pil_to_mask(self, pil_image):
        """将PIL图像转换为掩码tensor"""
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        mask_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(mask_array)

    def tensor2pil(self, image_tensor):
        """将tensor转换为PIL图像（符合ComfyUI格式）"""
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.cpu()
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.numpy()
        else:
            image_np = image_tensor
        if image_np.ndim == 3:
            image_np = (image_np * 255).astype(np.uint8)
        elif image_np.ndim == 4:
            image_np = (image_np[0] * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil2tensor(self, image):
        """将PIL图像转换为tensor（符合ComfyUI格式）"""
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32) / 255.0
        else:
            image_np = image.astype(np.float32) / 255.0
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)
        return torch.from_numpy(image_np)
