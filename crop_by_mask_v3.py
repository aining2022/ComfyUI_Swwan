import torch
from PIL import Image

from .layerstyle_utils import log, tensor2pil, pil2tensor, mask2image, image2mask, gaussian_blur, min_bounding_rect, max_inscribed_rect, mask_area
from .layerstyle_utils import num_round_up_to_multiple, draw_rect


class CropByMaskV3:

    def __init__(self):
        self.NODE_NAME = 'CropByMask V3'

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
            },
            "optional": {
                "image": ("IMAGE",),  # 可选的图像输入
                "mask_image": ("IMAGE",),  # 可选的遮罩图像输入（二值化图片）
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOX", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v3'
    CATEGORY = 'RunningHub/LayerUtility'

    def crop_by_mask_v3(self, detect,
                        top_reserve, bottom_reserve,
                        left_reserve, right_reserve, round_to_multiple,
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
        elif mask_image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
        else:
            raise ValueError("Cannot determine canvas size")

        ret_images = []
        ret_masks = []
        
        # 处理遮罩图像（二值化图片，白色为标记区域）
        if mask_image is not None:
            # 将 mask_image 转换为灰度图
            mask_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('L')
            
            # 创建预览图像
            preview_image = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
        else:
            # 如果没有 mask_image，创建一个全白的遮罩
            mask_pil = Image.new('L', (canvas_width, canvas_height), 255)
            preview_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # 检测裁剪框
        if crop_box is None:
            # 对遮罩进行模糊处理（用于某些检测模式）
            bluredmask = gaussian_blur(mask_pil, 20).convert('L')
            
            x = 0
            y = 0
            width = 0
            height = 0
            
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

            log(f"{self.NODE_NAME}: Box detected. x={x1}, y={y1}, width={x2-x1}, height={y2-y1}")
            crop_box = (x1, y1, x2, y2)
            
            # 在预览图上绘制检测到的区域（红色）
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#FF0000",
                                      line_width=max(1, (w + h) // 100))
        
        # 在预览图上绘制最终裁剪框（绿色）
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00FF00",
                                  line_width=max(1, (crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200))

        # 裁剪图像
        if image is not None:
            for i in range(image.shape[0]):
                img_pil = tensor2pil(torch.unsqueeze(image[i], 0)).convert('RGB')
                cropped_img = img_pil.crop(crop_box)
                ret_images.append(pil2tensor(cropped_img))
        else:
            # 如果没有输入图像，返回裁剪后的遮罩作为图像
            cropped_mask = mask_pil.crop(crop_box)
            ret_images.append(pil2tensor(cropped_mask.convert('RGB')))

        # 裁剪遮罩
        cropped_mask = mask_pil.crop(crop_box)
        ret_masks.append(pil2tensor(cropped_mask.convert('RGB')))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        
        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            list(crop_box),
            pil2tensor(preview_image),
        )


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask V3": CropByMaskV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask V3": "LayerUtility: CropByMask V3"
}
