"""
Bounded Image Crop Nodes
Migrated from WAS Node Suite
"""

import torch


class BoundedImageCrop:
    """
    Crops images using IMAGE_BOUNDS data.
    Support batch processing with one-to-one or one-to-many bounds-to-images mapping.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_bounds": ("IMAGE_BOUNDS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bounded_image_crop"
    CATEGORY = "Swwan/image"

    def bounded_image_crop(self, image, image_bounds):
        # Ensure we are working with batches
        image = image.unsqueeze(0) if image.dim() == 3 else image

        # If number of images and bounds don't match, then only the first bounds will be used
        # to crop the images, otherwise, each bounds will be used for each image 1 to 1
        bounds_len = 1 if len(image_bounds) != len(image) else len(image)

        cropped_images = []
        for idx in range(len(image)):
            # If only one bounds object, no need to extract and calculate more than once.
            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                rmin, rmax, cmin, cmax = image_bounds[idx]

                # Check if the provided bounds are valid
                if rmin > rmax or cmin > cmax:
                    raise ValueError("Invalid bounds provided. Please make sure the bounds are within the image dimensions.")

            cropped_images.append(image[idx][rmin:rmax+1, cmin:cmax+1, :])

        return (torch.stack(cropped_images, dim=0),)


class BoundedImageCropWithMask:
    """
    Crops images based on mask bounds with customizable padding.
    Returns both cropped images and the calculated IMAGE_BOUNDS.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_left": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_right": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_top": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
                "padding_bottom": ("INT", {"default": 64, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "return_list": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE_BOUNDS",)
    FUNCTION = "bounded_image_crop_with_mask"
    CATEGORY = "Swwan/image"

    def bounded_image_crop_with_mask(self, image, mask, padding_left, padding_right, padding_top, padding_bottom, return_list=False):
        # Ensure we are working with batches
        image = image.unsqueeze(0) if image.dim() == 3 else image
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask

        # If number of masks and images don't match, then only the first mask will be used on
        # the images, otherwise, each mask will be used for each image 1 to 1
        mask_len = 1 if len(image) != len(mask) else len(image)

        cropped_images = []
        all_bounds = []
        for i in range(len(image)):
            # Single mask or multiple?
            if (mask_len == 1 and i == 0) or mask_len > 0:
                rows = torch.any(mask[i], dim=1)
                cols = torch.any(mask[i], dim=0)
                rmin, rmax = torch.where(rows)[0][[0, -1]]
                cmin, cmax = torch.where(cols)[0][[0, -1]]

                rmin = max(rmin - padding_top, 0)
                rmax = min(rmax + padding_bottom, mask[i].shape[0] - 1)
                cmin = max(cmin - padding_left, 0)
                cmax = min(cmax + padding_right, mask[i].shape[1] - 1)

            # Even if only a single mask, create a bounds for each cropped image
            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin:rmax+1, cmin:cmax+1, :])

        if return_list:
            return cropped_images, all_bounds
        return torch.stack(cropped_images), all_bounds


NODE_CLASS_MAPPINGS = {
    "BoundedImageCrop": BoundedImageCrop,
    "BoundedImageCropWithMask": BoundedImageCropWithMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoundedImageCrop": "Bounded Image Crop (Swwan)",
    "BoundedImageCropWithMask": "Bounded Image Crop With Mask (Swwan)",
}
