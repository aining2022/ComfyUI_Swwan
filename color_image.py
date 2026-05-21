from PIL import ImageColor
import torch


class ColorImage:
    """Create a solid RGB image tensor without depending on ComfyUI_LayerStyle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "color": ("STRING", {"default": "#000000"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_image"
    CATEGORY = "😺dzNodes/LayerUtility"

    def color_image(self, width, height, color):
        rgb = _parse_color(color)
        values = torch.tensor(rgb, dtype=torch.float32).view(1, 1, 1, 3) / 255.0
        image = values.expand(1, int(height), int(width), 3).contiguous()
        return (image,)


def _parse_color(color):
    value = str(color).strip()
    if len(value) == 6 and all(char in "0123456789abcdefABCDEF" for char in value):
        value = f"#{value}"
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError as exc:
        raise ValueError(f"ColorImage expects a PIL-compatible color string, got {color!r}") from exc
    return tuple(rgb[:3])


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ColorImage": ColorImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ColorImage": "LayerUtility: ColorImage",
}
