# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import ProgressBar, model_management, torch

class ImageUpscaleWithModelBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "images": ("IMAGE",),
                              "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Same as ComfyUI native model upscaling node,
but allows setting sub-batches for reduced VRAM usage.
"""
    def upscale(self, upscale_model, images, per_batch):

        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = images.movedim(-1,-3)

        steps = in_img.shape[0]
        pbar = ProgressBar(steps)
        t = []

        for start_idx in range(0, in_img.shape[0], per_batch):
            sub_images = upscale_model(in_img[start_idx:start_idx+per_batch].to(device))
            t.append(sub_images.cpu())
            # Calculate the number of images processed in this batch
            batch_count = sub_images.shape[0]
            # Update the progress bar by the number of images processed in this batch
            pbar.update(batch_count)
        upscale_model.cpu()

        t = torch.cat(t, dim=0).permute(0, 2, 3, 1).cpu()

        return (t,)
