# SPDX-License-Identifier: MIT
# Frozen from custom_nodes/mask.py; see sources.json.
class MaskFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_dilate": ("INT", { "default": 0, "min": -256, "max": 256, "step": 1, }),
                "fill_holes": ("INT", { "default": 0, "min": 0, "max": 128, "step": 1, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "smooth": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, erode_dilate, smooth, remove_isolated_pixels, blur, fill_holes):
        masks = []
        for m in mask:
            # erode and dilate
            if erode_dilate != 0:
                if erode_dilate < 0:
                    m = torch.from_numpy(scipy.ndimage.grey_erosion(m.cpu().numpy(), size=(-erode_dilate, -erode_dilate)))
                else:
                    m = torch.from_numpy(scipy.ndimage.grey_dilation(m.cpu().numpy(), size=(erode_dilate, erode_dilate)))

            # fill holes
            if fill_holes > 0:
                #m = torch.from_numpy(scipy.ndimage.binary_fill_holes(m.cpu().numpy(), structure=np.ones((fill_holes,fill_holes)))).float()
                m = torch.from_numpy(scipy.ndimage.grey_closing(m.cpu().numpy(), size=(fill_holes, fill_holes)))

            # remove isolated pixels
            if remove_isolated_pixels > 0:
                m = torch.from_numpy(scipy.ndimage.grey_opening(m.cpu().numpy(), size=(remove_isolated_pixels, remove_isolated_pixels)))

            # smooth the mask
            if smooth > 0:
                if smooth % 2 == 0:
                    smooth += 1
                m = T.functional.gaussian_blur((m > 0.5).unsqueeze(0), smooth).squeeze(0)

            # blur the mask
            if blur > 0:
                if blur % 2 == 0:
                    blur += 1
                m = T.functional.gaussian_blur(m.float().unsqueeze(0), blur).squeeze(0)

            masks.append(m.float())

        masks = torch.stack(masks, dim=0).float()

        return (masks, )
