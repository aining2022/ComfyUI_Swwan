# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import ProgressBar, common_upscale, model_management, torch
from ..ops.transitions import crossfade, easing_functions, transition_images

class CrossFadeImages:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfadeimages"
    CATEGORY = "Swwan/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images_1": ("IMAGE",),
                 "images_2": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_start_index": ("INT", {"default": 1,"min": -4096, "max": 4096, "step": 1}),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "start_level": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "end_level": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01}),
        },
    }

    def crossfadeimages(self, images_1, images_2, transition_start_index, transitioning_frames, interpolation, start_level, end_level):

        crossfade_images = []

        if transition_start_index < 0:
            transition_start_index = len(images_1) + transition_start_index
            if transition_start_index < 0:
                raise ValueError("Transition start index is out of range for images_1.")

        transitioning_frames = min(transitioning_frames, len(images_1) - transition_start_index, len(images_2))

        alphas = torch.linspace(start_level, end_level, transitioning_frames)
        for i in range(transitioning_frames):
            alpha = alphas[i]
            image1 = images_1[transition_start_index + i]
            image2 = images_2[i]
            easing_function = easing_functions.get(interpolation)
            alpha = easing_function(alpha)  # Apply the easing function to the alpha value

            crossfade_image = crossfade(image1, image2, alpha)
            crossfade_images.append(crossfade_image)

        # Convert crossfade_images to tensor
        crossfade_images = torch.stack(crossfade_images, dim=0)

        # Append the beginning of images_1 (before the transition)
        beginning_images_1 = images_1[:transition_start_index]
        crossfade_images = torch.cat([beginning_images_1, crossfade_images], dim=0)

        # Append the remaining frames of images_2 (after the transition)
        remaining_images_2 = images_2[transitioning_frames:]
        if len(remaining_images_2) > 0:
            crossfade_images = torch.cat([crossfade_images, remaining_images_2], dim=0)

        return (crossfade_images, )

class CrossFadeImagesMulti:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfadeimages"
    CATEGORY = "Swwan/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                 "image_1": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
        "optional": {
            "image_2": ("IMAGE",),
        }
    }

    def crossfadeimages(self, inputcount, transitioning_frames, interpolation, **kwargs):

        image_1 = kwargs["image_1"]
        first_image_shape = image_1.shape
        first_image_device = image_1.device
        height = image_1.shape[1]
        width = image_1.shape[2]

        easing_function = easing_functions[interpolation]

        for c in range(1, inputcount):
            frames = []
            new_image = kwargs.get(f"image_{c + 1}", torch.zeros(first_image_shape)).to(first_image_device)
            new_image_height = new_image.shape[1]
            new_image_width = new_image.shape[2]

            if new_image_height != height or new_image_width != width:
                new_image = common_upscale(new_image.movedim(-1, 1), width, height, "lanczos", "disabled")
                new_image = new_image.movedim(1, -1)  # Move channels back to the last dimension

            last_frame_image_1 = image_1[-1]
            first_frame_image_2 = new_image[0]

            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=last_frame_image_1.dtype, device=last_frame_image_1.device)
                frame_image = crossfade(last_frame_image_1, first_frame_image_2, alpha_tensor)
                frames.append(frame_image)

            frames = torch.stack(frames)
            image_1 = torch.cat((image_1, frames, new_image), dim=0)

        return image_1,

class TransitionImagesMulti:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transition"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Creates transitions between images.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                 "image_1": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_type": (["horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade"],),
                 "transitioning_frames": ("INT", {"default": 2,"min": 2, "max": 4096, "step": 1}),
                 "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 100.0, "step": 0.1}),
                 "reverse": ("BOOLEAN", {"default": False}),
                 "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
    }

    def transition(self, inputcount, transitioning_frames, transition_type, interpolation, device, blur_radius, reverse, **kwargs):

        gpu = model_management.get_torch_device()

        image_1 = kwargs["image_1"]
        height = image_1.shape[1]
        width = image_1.shape[2]
        first_image_shape = image_1.shape
        first_image_device = image_1.device

        easing_function = easing_functions[interpolation]

        for c in range(1, inputcount):
            frames = []
            new_image = kwargs.get(f"image_{c + 1}", torch.zeros(first_image_shape)).to(first_image_device)
            new_image_height = new_image.shape[1]
            new_image_width = new_image.shape[2]

            if new_image_height != height or new_image_width != width:
                new_image = common_upscale(new_image.movedim(-1, 1), width, height, "lanczos", "disabled")
                new_image = new_image.movedim(1, -1)  # Move channels back to the last dimension

            last_frame_image_1 = image_1[-1]
            first_frame_image_2 = new_image[0]
            if device == "GPU":
                last_frame_image_1 = last_frame_image_1.to(gpu)
                first_frame_image_2 = first_frame_image_2.to(gpu)

            if reverse:
                last_frame_image_1, first_frame_image_2 = first_frame_image_2, last_frame_image_1

            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=last_frame_image_1.dtype, device=last_frame_image_1.device)
                frame_image = transition_images(last_frame_image_1, first_frame_image_2, alpha_tensor, transition_type, blur_radius, reverse)
                frames.append(frame_image)

            frames = torch.stack(frames).cpu()
            image_1 = torch.cat((image_1, frames, new_image), dim=0)

        return image_1.cpu(),

class TransitionImagesInBatch:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transition"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Creates transitions between images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                 "transition_type": (["horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade"],),
                 "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                 "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 100.0, "step": 0.1}),
                 "reverse": ("BOOLEAN", {"default": False}),
                 "device": (["CPU", "GPU"], {"default": "CPU"}),
        },
    }

    #transitions from matteo's essential nodes
    def transition(self, images, transitioning_frames, transition_type, interpolation, device, blur_radius, reverse):
        if images.shape[0] == 1:
            return images,

        gpu = model_management.get_torch_device()

        easing_function = easing_functions[interpolation]

        images_list = []
        pbar = ProgressBar(images.shape[0] - 1)
        for i in range(images.shape[0] - 1):
            frames = []
            image_1 = images[i]
            image_2 = images[i + 1]

            if device == "GPU":
                image_1 = image_1.to(gpu)
                image_2 = image_2.to(gpu)

            if reverse:
                image_1, image_2 = image_2, image_1

            for frame in range(transitioning_frames):
                t = frame / (transitioning_frames - 1)
                alpha = easing_function(t)
                alpha_tensor = torch.tensor(alpha, dtype=image_1.dtype, device=image_1.device)
                frame_image = transition_images(image_1, image_2, alpha_tensor, transition_type, blur_radius, reverse)
                frames.append(frame_image)
            pbar.update(1)

            frames = torch.stack(frames).cpu()
            images_list.append(frames)
        images = torch.cat(images_list, dim=0)

        return images.cpu(),

class ImageBatchJoinWithTransition:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transition_batches"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Transitions between two batches of images, starting at a specified index in the first batch.
During the transition, frames from both batches are blended frame-by-frame, so the video keeps playing.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "start_index": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
                "transition_type": (["horizontal slide", "vertical slide", "box", "circle", "horizontal door", "vertical door", "fade"],),
                "transitioning_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "reverse": ("BOOLEAN", {"default": False}),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
        }

    def transition_batches(self, images_1, images_2, start_index, interpolation, transition_type, transitioning_frames, blur_radius, reverse, device):
        if images_1.shape[0] == 0 or images_2.shape[0] == 0:
            raise ValueError("Both input batches must have at least one image.")

        if start_index < 0:
            start_index = images_1.shape[0] + start_index
        if start_index < 0 or start_index > images_1.shape[0]:
            raise ValueError("start_index is out of range.")

        gpu = model_management.get_torch_device()
        easing_function = easing_functions[interpolation]
        out_frames = []

        # Add images from images_1 up to start_index
        if start_index > 0:
            out_frames.append(images_1[:start_index])

        # Determine how many frames we can blend
        max_transition = min(transitioning_frames, images_1.shape[0] - start_index, images_2.shape[0])

        # Blend corresponding frames from both batches
        for i in range(max_transition):
            img1 = images_1[start_index + i]
            img2 = images_2[i]
            if device == "GPU":
                img1 = img1.to(gpu)
                img2 = img2.to(gpu)
            if reverse:
                img1, img2 = img2, img1
            t = i / (max_transition - 1) if max_transition > 1 else 1.0
            alpha = easing_function(t)
            alpha_tensor = torch.tensor(alpha, dtype=img1.dtype, device=img1.device)
            frame_image = transition_images(img1, img2, alpha_tensor, transition_type, blur_radius, reverse)
            out_frames.append(frame_image.cpu().unsqueeze(0))

        # Add remaining images from images_2 after transition
        if images_2.shape[0] > max_transition:
            out_frames.append(images_2[max_transition:])

        # Concatenate all frames
        out = torch.cat(out_frames, dim=0)
        return (out.cpu(),)

class ImageGridtoBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "columns": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "The number of columns in the grid."}),
                    "rows": ("INT", {"default": 0, "min": 1, "max": 8, "tooltip": "The number of rows in the grid. Set to 0 for automatic calculation."}),
                  }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decompose"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "Converts a grid of images to a batch of images."

    def decompose(self, image, columns, rows):
        B, H, W, C = image.shape
        print("input size: ", image.shape)

        # Calculate cell width, rounding down
        cell_width = W // columns

        if rows == 0:
            # If rows is 0, calculate number of full rows
            cell_height = H // columns
            rows = H // cell_height
        else:
            # If rows is specified, adjust cell_height
            cell_height = H // rows

        # Crop the image to fit full cells
        image = image[:, :rows*cell_height, :columns*cell_width, :]

        # Reshape and permute the image to get the grid
        image = image.view(B, rows, cell_height, columns, cell_width, C)
        image = image.permute(0, 1, 3, 2, 4, 5).contiguous()
        image = image.view(B, rows * columns, cell_height, cell_width, C)

        # Reshape to the final batch tensor
        img_tensor = image.view(-1, cell_height, cell_width, C)

        return (img_tensor,)
