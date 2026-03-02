"""Image Blend Node
Migrated from ComfyUI_LayerStyle/py/image_blend.py
Original author: chflame163
"""

import torch
from PIL import Image
from .layerstyle_utils import log, pil2tensor, tensor2pil, chop_image, chop_mode


class ImageBlendSwwan:

    def __init__(self):
        self.NODE_NAME = 'ImageBlendSwwan'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (chop_mode,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend'
    CATEGORY = "Swwan/image"

    def image_blend(self, background_image, layer_image,
                    invert_mask, blend_mode, opacity,
                    layer_mask=None):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []

        # Process background images
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))

        # Process layer images and extract alpha channel as mask
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))

        # Use provided mask if available
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        # Process each image in batch
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')

            # Blend layer with background
            _comp = chop_image(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "ImageBlendSwwan": ImageBlendSwwan
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendSwwan": "Image Blend (Swwan)"
}
