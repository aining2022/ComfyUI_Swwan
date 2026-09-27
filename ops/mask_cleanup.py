# SPDX-License-Identifier: MIT
# Derived from ComfyUI_essentials/mask.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import torch

class MaskCleanupAlgorithm:

    def execute(self, mask, erode_dilate, smooth, remove_isolated_pixels, blur, fill_holes):
        from scipy import ndimage
        from torchvision import transforms as T
        masks = []
        for m in mask:
            if erode_dilate != 0:
                if erode_dilate < 0:
                    m = torch.from_numpy(ndimage.grey_erosion(m.cpu().numpy(), size=(-erode_dilate, -erode_dilate)))
                else:
                    m = torch.from_numpy(ndimage.grey_dilation(m.cpu().numpy(), size=(erode_dilate, erode_dilate)))
            if fill_holes > 0:
                m = torch.from_numpy(ndimage.grey_closing(m.cpu().numpy(), size=(fill_holes, fill_holes)))
            if remove_isolated_pixels > 0:
                m = torch.from_numpy(ndimage.grey_opening(m.cpu().numpy(), size=(remove_isolated_pixels, remove_isolated_pixels)))
            if smooth > 0:
                if smooth % 2 == 0:
                    smooth += 1
                m = T.functional.gaussian_blur((m > 0.5).unsqueeze(0), smooth).squeeze(0)
            if blur > 0:
                if blur % 2 == 0:
                    blur += 1
                m = T.functional.gaussian_blur(m.float().unsqueeze(0), blur).squeeze(0)
            masks.append(m.float())
        masks = torch.stack(masks, dim=0).float()
        return (masks,)
