# SPDX-License-Identifier: GPL-3.0-only
# Derived from Goohaitools-comfyui/图像与遮罩预览.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os

class RegionPreviewAlgorithm:
    """
    图像与遮罩预览 孤海节点
    输入一张图像和遮罩，在图像上叠加遮罩区域颜色和序号
    """

    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB元组"""
        hex_color = hex_color.lstrip('#')
        return tuple((int(hex_color[i:i + 2], 16) for i in (0, 2, 4)))

    def convert_to_binary_mask(self, mask):
        """将输入遮罩转换为二值化遮罩"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = np.mean(mask, axis=2)
        if mask.max() > 1.0:
            mask = mask / 255.0
        binary_mask = (mask > 0.5).astype(np.float32)
        return binary_mask

    def get_mask_center_and_size(self, binary_mask):
        """计算遮罩的中心点和尺寸"""
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return (None, None, None, None)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        center_x = (cmin + cmax) // 2
        center_y = (rmin + rmax) // 2
        width = cmax - cmin
        height = rmax - rmin
        return (center_x, center_y, width, height)

    def get_text_metrics(self, draw, text, font):
        """获取文本的精确尺寸和基线信息"""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            baseline_offset = text_height * 0.75
            return (text_width, text_height, baseline_offset)
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
            baseline_offset = text_height * 0.75
            return (text_width, text_height, baseline_offset)

    def preview(self, 图像, 遮罩, 遮罩不透明, 遮罩颜色, 显示序号, 序号字体, 序号不透明, 序号缩放, 字号比例, 序号颜色):
        if 图像.dim() == 4:
            input_image = 图像[0]
        else:
            input_image = 图像
        img_np = input_image.cpu().numpy()
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3:4]
            rgb = img_np[:, :, :3]
            img_np = rgb * alpha
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = (img_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, mode='RGB')
        width, height = pil_image.size
        mask_list = []
        if 遮罩.dim() == 2:
            mask_list.append(self.convert_to_binary_mask(遮罩))
        elif 遮罩.dim() == 3:
            if 遮罩.shape[0] == 1:
                mask_list.append(self.convert_to_binary_mask(遮罩[0]))
            else:
                for i in range(遮罩.shape[0]):
                    mask_list.append(self.convert_to_binary_mask(遮罩[i]))
        else:
            mask_list.append(self.convert_to_binary_mask(遮罩))
        canvas = pil_image.copy()
        if len(mask_list) > 0:
            mask_color_rgb = self.hex_to_rgb(遮罩颜色)
            color_layer = Image.new('RGBA', (width, height), (*mask_color_rgb, 0))
            draw_color = ImageDraw.Draw(color_layer)
            for mask in mask_list:
                if mask is None or mask.size == 0:
                    continue
                if mask.shape[0] != height or mask.shape[1] != width:
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                    mask_img = mask_img.resize((width, height), Image.NEAREST)
                    mask = np.array(mask_img) / 255.0
                mask_alpha = (mask * 遮罩不透明 * 255).astype(np.uint8)
                alpha_layer = Image.fromarray(mask_alpha, mode='L')
                temp_canvas = Image.new('RGBA', (width, height))
                temp_canvas.paste(canvas, (0, 0))
                temp_canvas.paste(color_layer, (0, 0), alpha_layer)
                canvas = temp_canvas.convert('RGB')
        if 显示序号 and len(mask_list) > 0:
            number_color_rgb = self.hex_to_rgb(序号颜色)
            max_height = 0
            mask_info = []
            for idx, mask in enumerate(mask_list):
                if mask is None or mask.size == 0:
                    continue
                center_x, center_y, width_mask, height_mask = self.get_mask_center_and_size(mask)
                if center_x is None:
                    continue
                mask_info.append({'index': idx + 1, 'center': (center_x, center_y), 'height': height_mask})
                if 序号缩放 == '固定大小':
                    if height_mask > max_height:
                        max_height = height_mask
            for info in mask_info:
                idx = info['index']
                center_x, center_y = info['center']
                mask_height = info['height']
                if 序号缩放 == '跟随遮罩缩放':
                    font_size = int(mask_height * 字号比例)
                else:
                    font_size = int(max_height * 字号比例)
                font_size = max(1, font_size)
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    fonts_dir = str(Path(__file__).resolve().parents[1] / 'fonts')
                    font_path = os.path.join(fonts_dir, 序号字体)
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
                temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                text = str(idx)
                text_width, text_height, baseline_offset = self.get_text_metrics(temp_draw, text, font)
                draw_x = center_x - text_width // 2
                draw_y = center_y - baseline_offset
                temp_draw.text((draw_x, draw_y), text, fill=(*number_color_rgb, int(序号不透明 * 255)), font=font)
                canvas_rgba = canvas.convert('RGBA')
                combined = Image.alpha_composite(canvas_rgba, temp_img)
                canvas = combined.convert('RGB')
        output_array = np.array(canvas).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_array).unsqueeze(0)
        return (output_tensor,)
