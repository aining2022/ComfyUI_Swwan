# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import ThreadPoolExecutor, os, torch

class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [
                'mkl',
                'hm',
                'reinhard',
                'mvgd',
                'hm-mvgd-hm',
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Swwan/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic
color-grading of photographs, paintings and film sequences as well as light-field
and stopmotion corrections.

The methods behind the mappings are based on the approach from Reinhard et al.,
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.
https://github.com/hahnec/color-matcher/

"""

    def colormatch(self, image_ref, image_target, method, strength=1.0, multithread=True):
        # Skip unnecessary processing
        if strength == 0:
            return (image_target,)

        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)

        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        def process(i):
            cm = ColorMatcher()
            image_target_np_i = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method) # Avoid potential blur when only the fully color-matched image is used
                if strength != 1:
                    image_result = image_target_np_i + strength * (image_result - image_target_np_i)

                return torch.from_numpy(image_result)

            except Exception as e:
                print(f"Thread {i} error: {e}")
                return torch.from_numpy(image_target_np_i)  # fallback

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)

class ImageNormalize_Neg1_To_1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "images": ("IMAGE",),

                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Normalize the images to be in the range [-1, 1]
"""

    def normalize(self,images):
        images = images * 2.0 - 1.0
        return (images,)

class RemapImageRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
            "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            "clamp": ("BOOLEAN", {"default": True}),
            },
            }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Remaps the image values to the specified range.
"""

    def remap(self, image, min, max, clamp):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = min + image * (max - min)
        if clamp:
            image = torch.clamp(image, min=0.0, max=1.0)
        return (image, )

class SplitImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            },
            }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("red", "green", "blue", "mask")
    FUNCTION = "split"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Splits image channels into images where the selected channel
is repeated for all channels, and the alpha as a mask.
"""

    def split(self, image):
        red = image[:, :, :, 0:1] # Red channel
        green = image[:, :, :, 1:2] # Green channel
        blue = image[:, :, :, 2:3] # Blue channel
        alpha = image[:, :, :, 3:4] # Alpha channel
        alpha = alpha.squeeze(-1)

        # Repeat the selected channel for all channels
        red = torch.cat([red, red, red], dim=3)
        green = torch.cat([green, green, green], dim=3)
        blue = torch.cat([blue, blue, blue], dim=3)
        return (red, green, blue, alpha)

class MergeImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "red": ("IMAGE",),
            "green": ("IMAGE",),
            "blue": ("IMAGE",),

            },
            "optional": {
                "alpha": ("MASK", {"default": None}),
                },
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Merges channel data into an image.
"""

    def merge(self, red, green, blue, alpha=None):
        image = torch.stack([
        red[..., 0, None], # Red channel
        green[..., 1, None], # Green channel
        blue[..., 2, None]   # Blue channel
        ], dim=-1)
        image = image.squeeze(-2)
        if alpha is not None:
            image = torch.cat([image, alpha.unsqueeze(-1)], dim=-1)
        return (image,)
