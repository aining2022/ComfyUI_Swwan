# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import common_upscale, torch
from ..ops.grid import grid, fixed_grid

class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Concatenates the image2 to image1 in the specified direction.
"""

    def concatenate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2

            # Repeat the last image to match the largest batch size
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)

            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)

            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")

            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Ensure both images have the same number of channels
        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)


        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,

class ImageConcatFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "num_columns": ("INT", {"default": 3, "min": 1, "max": 255, "step": 1}),
            "match_image_size": ("BOOLEAN", {"default": False}),
            "max_resolution": ("INT", {"default": 4096}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
    Concatenates images from a batch into a grid with a specified number of columns.
    """

    def concat(self, images, num_columns, match_image_size, max_resolution):
        # Assuming images is a batch of images (B, H, W, C)
        batch_size, height, width, channels = images.shape
        num_rows = (batch_size + num_columns - 1) // num_columns  # Calculate number of rows

        print(f"Initial dimensions: batch_size={batch_size}, height={height}, width={width}, channels={channels}")
        print(f"num_rows={num_rows}, num_columns={num_columns}")

        if match_image_size:
            target_shape = images[0].shape

            resized_images = []
            for image in images:
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_aspect_ratio = original_width / original_height

                if original_aspect_ratio > 1:
                    target_height = target_shape[0]
                    target_width = int(target_height * original_aspect_ratio)
                else:
                    target_width = target_shape[1]
                    target_height = int(target_width / original_aspect_ratio)

                print(f"Resizing image from ({original_height}, {original_width}) to ({target_height}, {target_width})")

                # Resize the image to match the target size while preserving aspect ratio
                resized_image = common_upscale(image.movedim(-1, 0), target_width, target_height, "lanczos", "disabled")
                resized_image = resized_image.movedim(0, -1)  # Move channels back to the last dimension
                resized_images.append(resized_image)

            # Convert the list of resized images back to a tensor
            images = torch.stack(resized_images)

            height, width = target_shape[:2]  # Update height and width

        # Initialize an empty grid
        grid_height = num_rows * height
        grid_width = num_columns * width

        print(f"Grid dimensions before scaling: grid_height={grid_height}, grid_width={grid_width}")

        # Original scale factor calculation remains unchanged
        scale_factor = min(max_resolution / grid_height, max_resolution / grid_width, 1.0)

        # Apply scale factor to height and width
        scaled_height = height * scale_factor
        scaled_width = width * scale_factor

        # Round scaled dimensions to the nearest number divisible by 8
        height = max(1, int(round(scaled_height / 8) * 8))
        width = max(1, int(round(scaled_width / 8) * 8))

        if abs(scaled_height - height) > 4:
            height = max(1, int(round((scaled_height + 4) / 8) * 8))
        if abs(scaled_width - width) > 4:
            width = max(1, int(round((scaled_width + 4) / 8) * 8))

        # Recalculate grid dimensions with adjusted height and width
        grid_height = num_rows * height
        grid_width = num_columns * width
        print(f"Grid dimensions after scaling: grid_height={grid_height}, grid_width={grid_width}")
        print(f"Final image dimensions: height={height}, width={width}")

        grid = torch.zeros((grid_height, grid_width, channels), dtype=images.dtype)

        for idx, image in enumerate(images):
            resized_image = torch.nn.functional.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), size=(height, width), mode="bilinear").squeeze().permute(1, 2, 0)
            row = idx // num_columns
            col = idx % num_columns
            grid[row*height:(row+1)*height, col*width:(col+1)*width, :] = resized_image

        return grid.unsqueeze(0),

class ImageConcatMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),

                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "layout": (["strip", "grid", "batch_grid"], {"default": "strip"}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 1000}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Strip keeps the original concat behavior; grid uses multiple inputs; batch_grid lays out image_1 frames. inputcount updates ports automatically and preserves connected inputs.
"""

    def combine(self, inputcount, direction, match_image_size, layout="strip", columns=2, **kwargs):
        image = kwargs["image_1"]
        if layout == "batch_grid":
            return (grid([image[i:i+1] for i in range(image.shape[0])], columns, match_image_size),)
        if layout == "grid":
            tiles = [kwargs.get(f"image_{i+1}", torch.zeros_like(image)) for i in range(inputcount)]
            return (grid(tiles, columns, match_image_size),)
        first_image_shape = None
        if first_image_shape is None:
            first_image_shape = image.shape
        for c in range(1, inputcount):
            new_image = kwargs.get(f"image_{c + 1}", torch.zeros(first_image_shape))
            image, = ImageConcanate.concatenate(self, image, new_image, direction, match_image_size, first_image_shape=first_image_shape)
        first_image_shape = None
        return (image,)

class ImageGridComposite2x2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Concatenates the 4 input images into a 2x2 grid.
"""

    def compositegrid(self, image1, image2, image3, image4):
        return (fixed_grid([image1, image2, image3, image4], 2),)

class ImageGridComposite3x3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
            "image5": ("IMAGE",),
            "image6": ("IMAGE",),
            "image7": ("IMAGE",),
            "image8": ("IMAGE",),
            "image9": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Concatenates the 9 input images into a 3x3 grid.
"""

    def compositegrid(self, image1, image2, image3, image4, image5, image6, image7, image8, image9):
        return (fixed_grid([image1, image2, image3, image4, image5, image6, image7, image8, image9], 3),)
