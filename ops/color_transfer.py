# SPDX-License-Identifier: MIT
# Derived from ComfyUI_essentials/image.py; original authors and licenses in THIRD_PARTY_NOTICES.md.
# Modified: interface separated; optional imports deferred; FreeMono replaces unavailable fonts.
import torch
import comfy.utils
import comfy.model_management

class EssentialsColorAlgorithm:

    def execute(self, image, reference, color_space, factor, device, batch_size, reference_mask=None):
        import kornia
        if 'gpu' == device:
            device = comfy.model_management.get_torch_device()
        elif 'auto' == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'
        image = image.permute([0, 3, 1, 2])
        reference = reference.permute([0, 3, 1, 2]).to(device)
        if reference_mask is not None:
            assert reference_mask.ndim == 3, f'Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}'
            assert reference_mask.shape[0] == reference.shape[0], f'Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {reference.shape[0]}'
            reference_mask = reference_mask.unsqueeze(1).to(device)
            reference_mask = (reference_mask > 0.5).float()
            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(reference_mask, reference.shape[3], reference.shape[2], upscale_method='bicubic', crop='center')
        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]
        if 'LAB' == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif 'YCbCr' == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif 'LUV' == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif 'YUV' == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif 'XYZ' == color_space:
            reference = kornia.color.rgb_to_xyz(reference)
        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)
        image_batch = torch.split(image, batch_size, dim=0)
        output = []
        for image in image_batch:
            image = image.to(device)
            if color_space == 'LAB':
                image = kornia.color.rgb_to_lab(image)
            elif color_space == 'YCbCr':
                image = kornia.color.rgb_to_ycbcr(image)
            elif color_space == 'LUV':
                image = kornia.color.rgb_to_luv(image)
            elif color_space == 'YUV':
                image = kornia.color.rgb_to_yuv(image)
            elif color_space == 'XYZ':
                image = kornia.color.rgb_to_xyz(image)
            image_mean, image_std = self.compute_mean_std(image)
            matched = torch.nan_to_num((image - image_mean) / image_std) * torch.nan_to_num(reference_std) + reference_mean
            matched = factor * matched + (1 - factor) * image
            if color_space == 'LAB':
                matched = kornia.color.lab_to_rgb(matched)
            elif color_space == 'YCbCr':
                matched = kornia.color.ycbcr_to_rgb(matched)
            elif color_space == 'LUV':
                matched = kornia.color.luv_to_rgb(matched)
            elif color_space == 'YUV':
                matched = kornia.color.yuv_to_rgb(matched)
            elif color_space == 'XYZ':
                matched = kornia.color.xyz_to_rgb(matched)
            out = matched.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())
            output.append(out)
        out = None
        output = torch.cat(output, dim=0)
        return (output,)

    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            masked_tensor = tensor * mask
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)
            mask_sum = torch.clamp(mask_sum, min=1e-06)
            mean = torch.nan_to_num(masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum)
            std = torch.sqrt(torch.nan_to_num(((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / mask_sum))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return (mean, std)
