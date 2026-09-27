# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import F, MAX_RESOLUTION, common_upscale, model_management, torch, tqdm

class ImagePadForOutpaintMasked:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
        if mask is not None:
            if torch.allclose(mask, torch.zeros_like(mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    mask = None
        B, H, W, C = image.size()

        new_image = torch.ones(
            (B, H + top + bottom, W + left + right, C),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + H, left:left + W, :] = image

        if mask is None:
            new_mask = torch.ones(
                (B, H + top + bottom, W + left + right),
                dtype=torch.float32,
            )

            t = torch.zeros(
            (B, H, W),
            dtype=torch.float32
            )
        else:
            # If a mask is provided, pad it to fit the new image size
            mask = F.pad(mask, (left, right, top, bottom), mode='constant', value=0)
            mask = 1 - mask
            t = torch.zeros_like(mask)

        if feathering > 0 and feathering * 2 < H and feathering * 2 < W:

            for i in range(H):
                for j in range(W):
                    dt = i if top != 0 else H
                    db = H - i if bottom != 0 else H

                    dl = j if left != 0 else W
                    dr = W - j if right != 0 else W

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    if mask is None:
                        t[:, i, j] = v * v
                    else:
                        t[:, top + i, left + j] = v * v

        if mask is None:
            new_mask[:, top:top + H, left:left + W] = t
            return (new_image, new_mask,)
        else:
            return (new_image, mask,)

class ImagePadForOutpaintTargetSize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (s.upscale_methods,),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, target_width, target_height, feathering, upscale_method, mask=None):
        B, H, W, C = image.size()
        new_height = H
        new_width = W
         # Calculate the scaling factor while maintaining aspect ratio
        scaling_factor = min(target_width / W, target_height / H)

        # Check if the image needs to be downscaled
        if scaling_factor < 1:
            image = image.movedim(-1,1)
            # Calculate the new width and height after downscaling
            new_width = int(W * scaling_factor)
            new_height = int(H * scaling_factor)

            # Downscale the image
            image_scaled = common_upscale(image, new_width, new_height, upscale_method, "disabled").movedim(1,-1)
        else:
            # If downscaling is not needed, use the original image dimensions
            image_scaled = image

        # Ensure mask dimensions match image dimensions
        if mask is not None:
            mask_scaled = mask.unsqueeze(0)  # Add an extra dimension for batch size
            mask_scaled = F.interpolate(mask_scaled, size=(new_height, new_width), mode="nearest")
            mask_scaled = mask_scaled.squeeze(0)  # Remove the extra dimension after interpolation
        else:
            mask_scaled = None

        # Calculate how much padding is needed to reach the target dimensions
        pad_top = max(0, (target_height - new_height) // 2)
        pad_bottom = max(0, target_height - new_height - pad_top)
        pad_left = max(0, (target_width - new_width) // 2)
        pad_right = max(0, target_width - new_width - pad_left)

        # Now call the original expand_image with the calculated padding
        return ImagePadForOutpaintMasked.expand_image(self, image_scaled, pad_left, pad_top, pad_right, pad_bottom, feathering, mask_scaled)

class ImagePrepForICLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "output_width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "output_height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "border_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "latent_image": ("IMAGE",),
                "latent_mask": ("MASK",),
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, reference_image, output_width, output_height, border_width, latent_image=None, reference_mask=None, latent_mask=None):

        if reference_mask is not None:
            if torch.allclose(reference_mask, torch.zeros_like(reference_mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    reference_mask = None
        image = reference_image
        if latent_image is not None:
            if image.shape[0] != latent_image.shape[0]:
                image = image.repeat(latent_image.shape[0], 1, 1, 1)
        B, H, W, C = image.size()

        # Handle mask
        if reference_mask is not None:
            resized_mask = torch.nn.functional.interpolate(
                reference_mask.unsqueeze(1),
                size=(H, W),
                mode='nearest'
            ).squeeze(1)
            print(resized_mask.shape)
            image = image * resized_mask.unsqueeze(-1)

        # Calculate new width maintaining aspect ratio
        new_width = int((W / H) * output_height)

        # Resize image to new height while maintaining aspect ratio
        resized_image = common_upscale(image.movedim(-1,1), new_width, output_height, "lanczos", "disabled").movedim(1,-1)

        # Create padded image
        if latent_image is None:
            pad_image = torch.zeros((B, output_height, output_width, C), device=image.device)
        else:
            resized_latent_image = common_upscale(latent_image.movedim(-1,1), output_width, output_height, "lanczos", "disabled").movedim(1,-1)
            pad_image = resized_latent_image
            if latent_mask is not None:
                resized_latent_mask = torch.nn.functional.interpolate(
                    latent_mask.unsqueeze(1),
                    size=(pad_image.shape[1], pad_image.shape[2]),
                    mode='nearest'
                ).squeeze(1)

        if border_width > 0:
            border = torch.zeros((B, output_height, border_width, C), device=image.device)
            padded_image = torch.cat((resized_image, border, pad_image), dim=2)
            if latent_mask is not None:
                padded_mask = torch.zeros((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, (new_width + border_width):] = resized_latent_mask
            else:
                padded_mask = torch.ones((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, :new_width + border_width] = 0
        else:
            padded_image = torch.cat((resized_image, pad_image), dim=2)
            if latent_mask is not None:
                padded_mask = torch.zeros((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, new_width:] = resized_latent_mask
            else:
                padded_mask = torch.ones((B, padded_image.shape[1], padded_image.shape[2]), device=image.device)
                padded_mask[:, :, :new_width] = 0

        return (padded_image, padded_mask)

class ImageCropByMaskAndResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "base_resolution": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "padding": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "min_crop_resolution": ("INT", { "default": 128, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "max_crop_resolution": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),

            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", )
    RETURN_NAMES = ("images", "masks", "bbox",)
    FUNCTION = "crop"
    CATEGORY = "Swwan/image"

    def crop_by_mask(self, mask, padding=0, min_crop_resolution=None, max_crop_resolution=None):
        iy, ix = (mask == 1).nonzero(as_tuple=True)
        h0, w0 = mask.shape

        if iy.numel() == 0:
            x_c = w0 / 2.0
            y_c = h0 / 2.0
            width = 0
            height = 0
        else:
            x_min = ix.min().item()
            x_max = ix.max().item()
            y_min = iy.min().item()
            y_max = iy.max().item()

            width = x_max - x_min
            height = y_max - y_min

            if width > w0 or height > h0:
                raise Exception("Masked area out of bounds")

            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0

        if min_crop_resolution:
            width = max(width, min_crop_resolution)
            height = max(height, min_crop_resolution)

        if max_crop_resolution:
            width = min(width, max_crop_resolution)
            height = min(height, max_crop_resolution)

        if w0 <= width:
            x0 = 0
            w = w0
        else:
            x0 = max(0, x_c - width / 2 - padding)
            w = width + 2 * padding
            if x0 + w > w0:
                x0 = w0 - w

        if h0 <= height:
            y0 = 0
            h = h0
        else:
            y0 = max(0, y_c - height / 2 - padding)
            h = height + 2 * padding
            if y0 + h > h0:
                y0 = h0 - h

        return (int(x0), int(y0), int(w), int(h))

    def crop(self, image, mask, base_resolution, padding=0, min_crop_resolution=128, max_crop_resolution=512):
        mask = mask.round()
        image_list = []
        mask_list = []
        bbox_list = []

        # First, collect all bounding boxes
        bbox_params = []
        aspect_ratios = []
        for i in range(image.shape[0]):
            x0, y0, w, h = self.crop_by_mask(mask[i], padding, min_crop_resolution, max_crop_resolution)
            bbox_params.append((x0, y0, w, h))
            aspect_ratios.append(w / h)

        # Find maximum width and height
        max_w = max([w for x0, y0, w, h in bbox_params])
        max_h = max([h for x0, y0, w, h in bbox_params])
        max_aspect_ratio = max(aspect_ratios)

        # Ensure dimensions are divisible by 16
        max_w = (max_w + 15) // 16 * 16
        max_h = (max_h + 15) // 16 * 16
        # Calculate common target dimensions
        if max_aspect_ratio > 1:
            target_width = base_resolution
            target_height = int(base_resolution / max_aspect_ratio)
        else:
            target_height = base_resolution
            target_width = int(base_resolution * max_aspect_ratio)

        for i in range(image.shape[0]):
            x0, y0, w, h = bbox_params[i]

            # Adjust cropping to use maximum width and height
            x_center = x0 + w / 2
            y_center = y0 + h / 2

            x0_new = int(max(0, x_center - max_w / 2))
            y0_new = int(max(0, y_center - max_h / 2))
            x1_new = int(min(x0_new + max_w, image.shape[2]))
            y1_new = int(min(y0_new + max_h, image.shape[1]))
            x0_new = x1_new - max_w
            y0_new = y1_new - max_h

            cropped_image = image[i][y0_new:y1_new, x0_new:x1_new, :]
            cropped_mask = mask[i][y0_new:y1_new, x0_new:x1_new]

            # Ensure dimensions are divisible by 16
            target_width = (target_width + 15) // 16 * 16
            target_height = (target_height + 15) // 16 * 16

            cropped_image = cropped_image.unsqueeze(0).movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            cropped_image = common_upscale(cropped_image, target_width, target_height, "lanczos", "disabled")
            cropped_image = cropped_image.movedim(1, -1).squeeze(0)

            cropped_mask = cropped_mask.unsqueeze(0).unsqueeze(0)
            cropped_mask = common_upscale(cropped_mask, target_width, target_height, 'bilinear', "disabled")
            cropped_mask = cropped_mask.squeeze(0).squeeze(0)

            image_list.append(cropped_image)
            mask_list.append(cropped_mask)
            bbox_list.append((x0_new, y0_new, x1_new, y1_new))


        return (torch.stack(image_list), torch.stack(mask_list), bbox_list)

class ImageCropByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "crop"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "Crops the input images based on the provided mask."

    def crop(self, image, mask):
        B, H, W, C = image.shape
        mask = mask.round()

        # Find bounding box for each batch
        crops = []

        for b in range(B):
            # Get coordinates of non-zero elements
            rows = torch.any(mask[min(b, mask.shape[0]-1)] > 0, dim=1)
            cols = torch.any(mask[min(b, mask.shape[0]-1)] > 0, dim=0)
            if not rows.any() or not cols.any():
                crops.append(image[b:b+1])
                continue

            # Find boundaries
            y_min, y_max = torch.where(rows)[0][[0, -1]]
            x_min, x_max = torch.where(cols)[0][[0, -1]]

            # Crop image and mask
            crop = image[b:b+1, y_min:y_max+1, x_min:x_max+1, :]
            crops.append(crop)

        # Stack results back together
        if len({tuple(c.shape[1:]) for c in crops}) > 1:
            raise ValueError("Mask crops have different sizes; use Mask Crop or process images as a list.")
        cropped_images = torch.cat(crops, dim=0)

        return (cropped_images, )

class ImageUncropByMask:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "destination": ("IMAGE",),
                        "source": ("IMAGE",),
                        "bbox": ("BBOX",),
                     },
                "optional": {
                        "mask": ("MASK",),
                        "border_blending": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Blur the mask edges in the bounding box"}),
                     },
                }

    CATEGORY = "Swwan/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uncrop"

    def uncrop(self, destination, source, bbox, mask=None, border_blending=0):

        output_list = []

        B, H, W, C = destination.shape

        for i in range(source.shape[0]):
            x0, y0, x1, y1 = bbox[i]
            bbox_height = y1 - y0
            bbox_width = x1 - x0

            # Resize source image to match the bounding box dimensions
            #resized_source = F.interpolate(source[i].unsqueeze(0).movedim(-1, 1), size=(bbox_height, bbox_width), mode='bilinear', align_corners=False)
            resized_source = common_upscale(source[i].unsqueeze(0).movedim(-1, 1), bbox_width, bbox_height, "lanczos", "disabled")
            resized_source = resized_source.movedim(1, -1).squeeze(0)

            # Resize mask to match the bounding box dimensions
            if mask is not None:
                resized_mask = common_upscale(mask[i].unsqueeze(0).unsqueeze(0), bbox_width, bbox_height, "bilinear", "disabled")
                resized_mask = resized_mask.squeeze(0).squeeze(0)
            else:
                resized_mask = torch.ones((bbox_height, bbox_width), dtype=source.dtype, device=source.device)

            if border_blending > 0:
                # Apply gaussian blur to the mask
                sigma = float(border_blending)
                radius = max(1, int(3.0 * sigma))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=source.device, dtype=source.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * sigma * sigma))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)

                # (H, W) -> (1, 1, H, W)
                blurred_mask = resized_mask.unsqueeze(0).unsqueeze(0)
                blurred_mask = F.conv2d(blurred_mask, kx, padding=(0, radius), groups=1)
                blurred_mask = F.conv2d(blurred_mask, ky, padding=(radius, 0), groups=1)
                resized_mask = blurred_mask.squeeze(0).squeeze(0)

            # Calculate padding values
            pad_left = x0
            pad_right = W - x1
            pad_top = y0
            pad_bottom = H - y1

            # Pad the resized source image and mask to fit the destination dimensions
            padded_source = F.pad(resized_source, pad=(0, 0, pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            padded_mask = F.pad(resized_mask, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

            # Ensure the padded mask has the correct shape
            padded_mask = padded_mask.unsqueeze(2).expand(-1, -1, destination[i].shape[2])
            # Ensure the padded source has the correct shape
            padded_source = padded_source.unsqueeze(2).expand(-1, -1, -1, destination[i].shape[2]).squeeze(2)

            # Combine the destination and padded source images using the mask
            result = destination[i] * (1.0 - padded_mask) + padded_source * padded_mask

            output_list.append(result)


        return (torch.stack(output_list),)

class ImageCropByMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "masks": ("MASK", ),
                    "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, }),
                    "preserve_size": ("BOOLEAN", {"default": False}),
                    "bg_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
                  }
                }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "crop"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "Crops the input images based on the provided masks."

    def crop(self, image, masks, width, height, bg_color, padding, preserve_size):
        B, H, W, C = image.shape
        BM, HM, WM = masks.shape
        mask_count = BM
        if HM != H or WM != W:
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)
            print(masks.shape)
        output_images = []
        output_masks = []

        bg_color = [int(x.strip())/255.0 for x in bg_color.split(",")]

        # For each mask
        for i in range(mask_count):
            curr_mask = masks[i]

            # Find bounds
            y_indices, x_indices = torch.nonzero(curr_mask, as_tuple=True)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            # Get exact bounds with padding
            min_y = max(0, y_indices.min().item() - padding)
            max_y = min(H, y_indices.max().item() + 1 + padding)
            min_x = max(0, x_indices.min().item() - padding)
            max_x = min(W, x_indices.max().item() + 1 + padding)

            # Ensure mask has correct shape for multiplication
            curr_mask = curr_mask.unsqueeze(-1).expand(-1, -1, C)

            # Crop image and mask together
            cropped_img = image[0, min_y:max_y, min_x:max_x, :]
            cropped_mask = curr_mask[min_y:max_y, min_x:max_x, :]

            crop_h, crop_w = cropped_img.shape[0:2]
            new_w = crop_w
            new_h = crop_h

            if not preserve_size or crop_w > width or crop_h > height:
                scale = min(width/crop_w, height/crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)

                # Resize RGB
                resized_img = common_upscale(cropped_img.permute(2,0,1).unsqueeze(0), new_w, new_h, "lanczos", "disabled").squeeze(0).permute(1,2,0)
                resized_mask = torch.nn.functional.interpolate(
                    cropped_mask.permute(2,0,1).unsqueeze(0),
                    size=(new_h, new_w),
                    mode='nearest'
                ).squeeze(0).permute(1,2,0)
            else:
                resized_img = cropped_img
                resized_mask = cropped_mask

            # Create empty tensors
            new_img = torch.zeros((height, width, 3), dtype=image.dtype)
            new_mask = torch.zeros((height, width), dtype=image.dtype)

            # Pad both
            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2
            new_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized_img
            if len(resized_mask.shape) == 3:
                resized_mask = resized_mask[:,:,0]  # Take first channel if 3D
            new_mask[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_mask

            output_images.append(new_img)
            output_masks.append(new_mask)

        if not output_images:
            return (image.new_zeros((0, height, width, 3)), masks.new_zeros((0, height, width)))

        out_rgb = torch.stack(output_images, dim=0)
        out_masks = torch.stack(output_masks, dim=0)

        # Apply mask to RGB
        mask_expanded = out_masks.unsqueeze(-1).expand(-1, -1, -1, 3)
        background_color = torch.tensor(bg_color, dtype=torch.float32, device=image.device)
        out_rgb = out_rgb * mask_expanded + background_color * (1 - mask_expanded)

        return (out_rgb, out_masks)

class ImagePadKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "extra_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "pad_mode": (["edge", "edge_pixel", "color", "pillarbox_blur"],),
                    "color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
                  },
                "optional": {
                    "mask": ("MASK", ),
                    "target_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                    "target_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                }
                }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "Pad the input image and optionally mask with the specified padding."

    def pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
                if pad_mode == "edge":
                    # Pad with edge color (mean)
                    top_edge = image[b, 0, :, :]
                    bottom_edge = image[b, H-1, :, :]
                    left_edge = image[b, :, 0, :]
                    right_edge = image[b, :, W-1, :]
                    out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                    out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                    out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                    out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                elif pad_mode == "edge_pixel":
                    # Pad with exact edge pixel values
                    for y in range(pad_top):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                    for y in range(pad_top+H, padded_height):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                    for x in range(pad_left):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                    for x in range(pad_left+W, padded_width):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                    out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                    out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                    out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                    out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                else:
                    # Pad with specified background color
                    out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)

class DrawMaskOnImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255 or 0.0-1.0, separated by commas."}),
                    "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                  },
                  "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Device to use for processing"}),
                }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "Swwan/masking"
    DESCRIPTION = "Applies the provided masks to the input images."

    def apply(self, image, mask, color, opacity=1.0, device="cpu"):
        B, H, W, C = image.shape
        BM, HM, WM = mask.shape

        processing_device = model_management.get_torch_device() if device == "gpu" else torch.device("cpu")

        in_masks = mask.clone().to(processing_device)
        in_images = image.clone().to(processing_device)

        if HM != H or WM != W:
            in_masks = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)
        if B > BM:
            in_masks = in_masks.repeat((B + BM - 1) // BM, 1, 1)[:B]
        elif BM > B:
            in_masks = in_masks[:B]

        output_images = []

        # Parse background color - detect if values are integers or floats
        bg_values = []
        for x in color.split(","):
            val_str = x.strip()
            if '.' in val_str:
                bg_values.append(float(val_str))
            else:
                bg_values.append(int(val_str) / 255.0)

        background_color = torch.tensor(bg_values, dtype=torch.float32, device=in_images.device)

        for i in tqdm(range(B), desc="DrawMaskOnImage batch"):
            curr_mask = in_masks[i]
            img_idx = min(i, B - 1)
            curr_image = in_images[img_idx]
            # Apply opacity to the mask: effective_mask = mask * opacity
            effective_mask = curr_mask * opacity
            mask_expanded = effective_mask.unsqueeze(-1).expand(-1, -1, 3)
            # Use strict interpolation formula: Image * (1 - Alpha) + Color * Alpha
            masked_image = curr_image * (1 - mask_expanded) + background_color * (mask_expanded)
            output_images.append(masked_image)

        # If no masks were processed, return empty tensor
        if not output_images:
            return (torch.zeros((0, H, W, 3), dtype=image.dtype),)

        out_rgb = torch.stack(output_images, dim=0).cpu()

        return (out_rgb, )
