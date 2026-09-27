# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import common_upscale, torch
from ..ops.transitions import crossfade, ease_in_out

class ImagePass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Passes the image through without modifying it.
"""

    def passthrough(self, image=None):
        return image,

class GetImageSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE","INT", "INT", "INT",)
    RETURN_NAMES = ("image", "width", "height", "count",)
    FUNCTION = "getsize"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Returns width, height and batch size of the image,
and passes it through unchanged.

"""

    def getsize(self, image):
        width = image.shape[2]
        height = image.shape[1]
        count = image.shape[0]
        return {"ui": {
            "text": [f"{count}x{width}x{height}"]},
            "result": (image, width, height, count)
        }

class GetLatentSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent": ("LATENT",),
        }}

    RETURN_TYPES = ("LATENT","INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "batch_size", "channels", "frames", "width", "height")
    FUNCTION = "getsize"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Returns latent tensor dimensions,
and passes the latent through unchanged.

"""
    def getsize(self, latent):
        if len(latent["samples"].shape) == 5:
            B, C, T, H, W = latent["samples"].shape
        elif len(latent["samples"].shape) == 4:
            B, C, H, W = latent["samples"].shape
            T = 0
        else:
            raise ValueError("Invalid latent shape")

        return {"ui": {
            "text": [f"{B}x{C}x{T}x{H}x{W}"]},
            "result": (latent, B, C, T, H, W)
        }

class ImageBatchRepeatInterleaving:

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "repeat"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Repeats each image in a batch by the specified number of times.
Example batch of 5 images: 0, 1 ,2, 3, 4
with repeats 2 becomes batch of 10 images: 0, 0, 1, 1, 2, 2, 3, 3, 4, 4
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def repeat(self, images, repeats, mask=None):
        original_count = images.shape[0]
        total_count = original_count * repeats

        repeated_images = torch.repeat_interleave(images, repeats=repeats, dim=0)
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=repeats, dim=0)
        else:
            mask = torch.zeros((total_count, images.shape[1], images.shape[2]),
                              device=images.device, dtype=images.dtype)
            for i in range(original_count):
                mask[i * repeats] = 1.0

        print("mask shape", mask.shape)
        return (repeated_images, mask)

class ShuffleImageBatch:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shuffle"
    CATEGORY = "Swwan/image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
        },
    }

    def shuffle(self, images, seed):
        torch.manual_seed(seed)
        B, H, W, C = images.shape
        indices = torch.randperm(B)
        shuffled_images = images[indices]

        return shuffled_images,

class GetImageRangeFromBatch:

    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Returns a range of images from a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
        },
        "optional": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
        }
    }

    def imagesfrombatch(self, start_index, num_frames, images=None, masks=None):
        chosen_images = None
        chosen_masks = None

        # Process images if provided
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            if start_index < 0 or start_index >= len(images):
                raise ValueError("Start index is out of range")
            end_index = min(start_index + num_frames, len(images))
            chosen_images = images[start_index:end_index]

        # Process masks if provided
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            if start_index < 0 or start_index >= len(masks):
                raise ValueError("Start index is out of range for masks")
            end_index = min(start_index + num_frames, len(masks))
            chosen_masks = masks[start_index:end_index]

        return (chosen_images, chosen_masks,)

class ImageBatchExtendWithOverlap:

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("source_images", "start_images", "extended_images")
    OUTPUT_TOOLTIPS = (
        "The original source images (passthrough)",
        "The input images used as the starting point for extension",
        "The extended images with overlap, if no new images are provided this will be empty",
    )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Helper node for video generation extension
First input source and overlap amount to get the starting frames for the extension.
Then on another copy of the node provide the newly generated frames and choose how to overlap them.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_images": ("IMAGE", {"tooltip": "The source images to extend"}),
                "overlap": ("INT", {"default": 13,"min": 1, "max": 4096, "step": 1, "tooltip": "Number of overlapping frames between source and new images"}),
                "overlap_side": (["source", "new_images"], {"default": "source", "tooltip": "Which side to overlap on"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out"], {"default": "linear_blend", "tooltip": "Method to use for overlapping frames"}),
        },
        "optional": {
            "new_images": ("IMAGE", {"tooltip": "The new images to extend with"}),
        }
    }

    def imagesfrombatch(self, source_images, overlap, overlap_side, overlap_mode, new_images=None):
        if overlap >= len(source_images):
            return source_images, source_images, source_images

        if new_images is not None:
            if source_images.shape[1:3] != new_images.shape[1:3]:
                raise ValueError(f"Source and new images must have the same shape: {source_images.shape[1:3]} vs {new_images.shape[1:3]}")
            # Determine where to place the overlap
            prefix = source_images[:-overlap]
            if overlap_side == "source":
                blend_src = source_images[-overlap:]
                blend_dst = new_images[:overlap]
            elif overlap_side == "new_images":
                blend_src = new_images[:overlap]
                blend_dst = source_images[-overlap:]
            suffix = new_images[overlap:]

            if overlap_mode == "linear_blend":
                blended_images = [
                    crossfade(blend_src[i], blend_dst[i], (i + 1) / (overlap + 1))
                    for i in range(overlap)
                ]
                blended_images = torch.stack(blended_images, dim=0)
                extended_images = torch.cat((prefix, blended_images, suffix), dim=0)
            elif overlap_mode == "ease_in_out":
                blended_images = []
                for i in range(overlap):
                    t = (i + 1) / (overlap + 1)
                    eased_t = ease_in_out(t)
                    blended_image = crossfade(blend_src[i], blend_dst[i], eased_t)
                    blended_images.append(blended_image)
                blended_images = torch.stack(blended_images, dim=0)
                extended_images = torch.cat((prefix, blended_images, suffix), dim=0)

            elif overlap_mode == "cut":
                extended_images = torch.cat((prefix, suffix), dim=0)
                if overlap_side == "new_images":
                   extended_images = torch.cat((source_images, new_images[overlap:]), dim=0)
                elif overlap_side == "source":
                   extended_images = torch.cat((source_images[:-overlap], new_images), dim=0)
        else:
            extended_images = torch.zeros((1, 64, 64, 3), device="cpu")

        start_images = source_images[-overlap:]

        return (source_images, start_images, extended_images)

class GetLatentRangeFromBatch:

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "latentsfrombatch"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Returns a range of latents from a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1,"min": -1, "max": 4096, "step": 1}),
        },
    }

    def latentsfrombatch(self, latents, start_index, num_frames):
        chosen_latents = None
        samples = latents["samples"]
        if len(samples.shape) == 4:
            B, C, H, W = samples.shape
            num_latents = B
        elif len(samples.shape) == 5:
            B, C, T, H, W = samples.shape
            num_latents = T

        if start_index == -1:
            start_index = max(0, num_latents - num_frames)
        if start_index < 0 or start_index >= num_latents:
            raise ValueError("Start index is out of range")

        end_index = num_latents if num_frames == -1 else min(start_index + num_frames, num_latents)

        if len(samples.shape) == 4:
            chosen_latents = samples[start_index:end_index]
        elif len(samples.shape) == 5:
            chosen_latents = samples[:, :, start_index:end_index]

        return ({"samples": chosen_latents.contiguous(),},)

class InsertLatentToIndex:

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "insert"
    CATEGORY = "KJNodes/latents"
    DESCRIPTION = """
Inserts a latent at the specified index into the original latent batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("LATENT",),
                "destination": ("LATENT",),
                "index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
        },
    }

    def insert(self, source, destination, index):
        samples_destination = destination["samples"]
        samples_source = source["samples"].to(samples_destination)

        if len(samples_source.shape) == 4:
            B, C, H, W = samples_source.shape
            num_latents = B
        elif len(samples_source.shape) == 5:
            B, C, T, H, W = samples_source.shape
            num_latents = T

        if index >= num_latents or index < 0:
            raise ValueError(f"Index {index} out of bounds for tensor with {num_latents} latents")

        if len(samples_source.shape) == 4:
            joined_latents = torch.cat([
                samples_destination[:index],
                samples_source,
                samples_destination[index+1:]
            ], dim=0)
        else:
            joined_latents = torch.cat([
                samples_destination[:, :, :index],
                samples_source,
                samples_destination[:, :, index+1:]
            ], dim=2)

        return ({"samples": joined_latents,},)

class ImageBatchFilter:

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "removed_indices",)
    FUNCTION = "filter"
    CATEGORY = "Swwan/image"
    DESCRIPTION = "Removes empty images from a batch"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "empty_color": ("STRING", {"default": "0, 0, 0"}),
                "empty_threshold": ("FLOAT", {"default": 0.01,"min": 0.0, "max": 1.0, "step": 0.01}),
        },
        "optional": {
            "replacement_image": ("IMAGE",),
        }
    }

    def filter(self, images, empty_color, empty_threshold, replacement_image=None):
        B, H, W, C = images.shape

        input_images = images.clone()

        empty_color_list = [int(color.strip()) for color in empty_color.split(',')]
        empty_color_tensor = torch.tensor(empty_color_list, dtype=torch.float32).to(input_images.device)

        color_diff = torch.abs(input_images - empty_color_tensor)
        mean_diff = color_diff.mean(dim=(1, 2, 3))

        empty_indices = mean_diff <= empty_threshold
        empty_indices_string = ', '.join([str(i) for i in range(B) if empty_indices[i]])

        if replacement_image is not None:
            B_rep, H_rep, W_rep, C_rep = replacement_image.shape
            replacement = replacement_image.clone()
            if (H_rep != images.shape[1]) or (W_rep != images.shape[2]) or (C_rep != images.shape[3]):
                replacement = common_upscale(replacement.movedim(-1, 1), W, H, "lanczos", "center").movedim(1, -1)
            input_images[empty_indices] = replacement[0]

            return (input_images, empty_indices_string,)
        else:
            non_empty_images = input_images[~empty_indices]
            return (non_empty_images, empty_indices_string,)

class GetImagesFromBatchIndexed:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Selects and returns the images at the specified indices as an image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
        },
    }

    def indexedimagesfrombatch(self, images, indexes):

        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]

        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)

        # Select the images at the specified indices
        chosen_images = images[indices_tensor]

        return (chosen_images,)

class InsertImagesToBatchIndexed:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "insertimagesfrombatch"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Inserts images at the specified indices into the original image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "images_to_insert": ("IMAGE",),
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
            "optional": {
                "mode": (["replace", "insert"],),
            }
        }

    def insertimagesfrombatch(self, original_images, images_to_insert, indexes, mode="replace"):
        if indexes == "":
            return (original_images,)

        input_images = original_images.clone()

        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]

        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)

        # Ensure the images_to_insert is a tensor
        if not isinstance(images_to_insert, torch.Tensor):
            images_to_insert = torch.tensor(images_to_insert)

        if mode == "replace":
            # Replace the images at the specified indices
            for index, image in zip(indices_tensor, images_to_insert):
                input_images[index] = image
        else:
            # Create a list to hold the new image sequence
            new_images = []
            insert_offset = 0

            for i in range(len(input_images) + len(indices_tensor)):
                if insert_offset < len(indices_tensor) and i == indices_tensor[insert_offset]:
                    # Use modulo to cycle through images_to_insert
                    new_images.append(images_to_insert[insert_offset % len(images_to_insert)])
                    insert_offset += 1
                else:
                    new_images.append(input_images[i - insert_offset])

            # Convert the list back to a tensor
            input_images = torch.stack(new_images, dim=0)

        return (input_images,)

class PadImageBatchInterleaved:

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Inserts empty frames between the images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "empty_frames_per_image": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
                "pad_frame_value": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                "add_after_last": ("BOOLEAN", {"default": False}),
            },
        }

    def pad(self, images, empty_frames_per_image, pad_frame_value, add_after_last):
        B, H, W, C = images.shape

        # Handle single frame case specifically
        if B == 1:
            total_frames = 1 + empty_frames_per_image if add_after_last else 1
        else:
            # Original B images + (B-1) sets of empty frames between them
            total_frames = B + (B-1) * empty_frames_per_image
            # Add additional empty frames after the last image if requested
            if add_after_last:
                total_frames += empty_frames_per_image

        # Create new tensor with zeros (empty frames)
        padded_batch = torch.ones((total_frames, H, W, C),
                                dtype=images.dtype,
                                device=images.device) * pad_frame_value
        # Create mask tensor (1 for original frames, 0 for empty frames)
        mask = torch.zeros((total_frames, H, W),
                        dtype=images.dtype,
                        device=images.device)

        # Fill in original images at their new positions
        for i in range(B):
            if B == 1:
                # For single frame, just place it at the beginning
                new_pos = 0
            else:
                # Each image is separated by empty_frames_per_image blank frames
                new_pos = i * (empty_frames_per_image + 1)

            padded_batch[new_pos] = images[i]
            mask[new_pos] = 1.0  # Mark this as an original frame

        return (padded_batch, mask)

class ReplaceImagesInBatch:

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "replace"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Replaces the images in a batch, starting from the specified start index,
with the replacement images.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
        "optional": {
            "original_images": ("IMAGE",),
            "replacement_images": ("IMAGE",),
            "original_masks": ("MASK",),
            "replacement_masks": ("MASK",),
        }
    }

    def replace(self, original_images=None, replacement_images=None, start_index=1, original_masks=None, replacement_masks=None):
        images = None
        masks = None

        if original_images is not None and replacement_images is not None:
            if start_index >= len(original_images):
                raise ValueError("ReplaceImagesInBatch: Start index is out of range")
            end_index = start_index + len(replacement_images)
            if end_index > len(original_images):
                raise ValueError("ReplaceImagesInBatch: End index is out of range")

            original_images_copy = original_images.clone()
            if original_images_copy.shape[2] != replacement_images.shape[2] or original_images_copy.shape[3] != replacement_images.shape[3]:
                replacement_images = common_upscale(replacement_images.movedim(-1, 1), original_images_copy.shape[1], original_images_copy.shape[2], "lanczos", "center").movedim(1, -1)

            original_images_copy[start_index:end_index] = replacement_images
            images = original_images_copy
        else:
            images = torch.zeros((1, 64, 64, 3))

        if original_masks is not None and replacement_masks is not None:
            if start_index >= len(original_masks):
                raise ValueError("ReplaceImagesInBatch: Start index is out of range")
            end_index = start_index + len(replacement_masks)
            if end_index > len(original_masks):
                raise ValueError("ReplaceImagesInBatch: End index is out of range")

            original_masks_copy = original_masks.clone()
            if original_masks_copy.shape[1] != replacement_masks.shape[1] or original_masks_copy.shape[2] != replacement_masks.shape[2]:
                replacement_masks = common_upscale(replacement_masks.unsqueeze(1), original_masks_copy.shape[1], original_masks_copy.shape[2], "nearest-exact", "center").squeeze(0)

            original_masks_copy[start_index:end_index] = replacement_masks
            masks = original_masks_copy
        else:
            masks = torch.zeros((1, 64, 64))

        return (images, masks)

class ReverseImageBatch:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverseimagebatch"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Reverses the order of the images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
        },
    }

    def reverseimagebatch(self, images):
        reversed_images = torch.flip(images, [0])
        return (reversed_images, )

class ImageBatchMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),

            },
            "optional": {
                "image_2": ("IMAGE", ),
            }
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Creates an image batch from multiple images.
You can set how many inputs the node has,
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, **kwargs):
        from nodes import ImageBatch
        image_batch_node = ImageBatch()
        image = kwargs["image_1"].cpu()
        first_image_shape = image.shape
        for c in range(1, inputcount):
            new_image = kwargs.get(f"image_{c + 1}", torch.zeros(first_image_shape)).cpu()
            image, = image_batch_node.batch(image, new_image)
        return (image,)

class ImageTensorList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Creates an image list from the input images.
"""

    def append(self, image1, image2):
        image_list = []
        if isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor):
            image_list = [image1, image2]
        elif isinstance(image1, list) and isinstance(image2, torch.Tensor):
            image_list = image1 + [image2]
        elif isinstance(image1, torch.Tensor) and isinstance(image2, list):
            image_list = [image1] + image2
        elif isinstance(image1, list) and isinstance(image2, list):
            image_list = image1 + image2
        return image_list,

class ImageAddMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "blending": (
                [   'add',
                    'subtract',
                    'multiply',
                    'difference',
                ],
                {
                "default": 'add'
                }),
                "blend_amount": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "add"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Add blends multiple images together.
You can set how many inputs the node has,
with the **inputcount** and clicking update.
"""

    def add(self, inputcount, blending, blend_amount, **kwargs):
        image = kwargs["image_1"]
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            if blending == "add":
                image = torch.add(image * blend_amount, new_image * blend_amount)
            elif blending == "subtract":
                image = torch.sub(image * blend_amount, new_image * blend_amount)
            elif blending == "multiply":
                image = torch.mul(image * blend_amount, new_image * blend_amount)
            elif blending == "difference":
                image = torch.sub(image, new_image)
        return (image,)
