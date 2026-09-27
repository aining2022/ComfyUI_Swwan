# SPDX-License-Identifier: GPL-3.0-only
# Derived from ComfyUI-KJNodes/mask_nodes.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import numpy as np
import torch
from PIL import ImageFilter
import comfy.utils
import comfy.model_management
from ..layerstyle_utils import tensor2pil, pil2tensor
main_device = comfy.model_management.get_torch_device()

def tqdm(values, **kwargs):
    return values

class GrowMaskAlgorithm:

    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        from scipy import ndimage
        import kornia.morphology as morph
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        previous_output = None
        current_expand = expand
        for m in tqdm(growmask, desc='Expanding/Contracting Mask'):
            output = m.unsqueeze(0).unsqueeze(0).to(main_device)
            if abs(round(current_expand)) > 0 and output.max() > 0:
                if tapered_corners:
                    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=output.device)
                else:
                    kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32, device=output.device)
                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = morph.erosion(output, kernel)
                    else:
                        output = morph.dilation(output, kernel)
            output = output.squeeze(0).squeeze(0)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = ndimage.binary_fill_holes(output_np)
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)
            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output.cpu())
        if blur_radius != 0:
            for idx, tensor in enumerate(out):
                pil_image = tensor2pil(tensor.cpu().detach())
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0))
