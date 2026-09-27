# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import BytesIO, Image, ImageDraw, ImageFont, ImageOps, MAX_RESOLUTION, PngInfo, ProgressBar, SaveImage, args, base64, composite, folder_paths, importlib, json, node_helpers, np, os, random, re, torch
import pathlib
import hashlib
from ..ops.image_save import _build_png_metadata, _save_image_with_fallback, _legacy_output_folder
from .resize import ImageResizeKJv2

class SaveImageWithAlpha:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_alpha"
    OUTPUT_NODE = True
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Saves an image and mask as .PNG with the mask as the alpha channel.
"""

    def save_images_alpha(self, images, mask, filename_prefix="ComfyUI_image_with_alpha", prompt=None, extra_pnginfo=None):
        from PIL.PngImagePlugin import PngInfo
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        if mask.dtype == torch.float16:
            mask = mask.to(torch.float32)
        def file_counter():
            max_counter = 0
            # Loop through the existing files
            for existing_file in sorted(os.listdir(full_output_folder)):
                # Check if the file matches the expected format
                match = re.fullmatch(fr"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter
            return max_counter

        for image, alpha in zip(images, mask):
            i = 255. * image.cpu().numpy()
            a = 255. * alpha.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

             # Resize the mask to match the image size
            a_resized = Image.fromarray(a).resize(img.size, Image.LANCZOS)
            a_resized = np.clip(a_resized, 0, 255).astype(np.uint8)
            img.putalpha(Image.fromarray(a_resized, mode='L'))
            metadata = _build_png_metadata(prompt, extra_pnginfo)

            # Increment the counter by 1 to get the next available value
            counter = file_counter() + 1
            file = f"{filename}_{counter:05}.png"
            _save_image_with_fallback(img, os.path.join(full_output_folder, file), {"format":"PNG", "pnginfo":metadata, "compress_level":4})
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

        return { "ui": { "images": results } }

class AddLabel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image":("IMAGE",),
            "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 48, "min": -1, "max": 4096, "step": 1}),
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
            "font_color": ("STRING", {"default": "white"}),
            "label_color": ("STRING", {"default": "black"}),
            "font": ((folder_paths.get_filename_list("swwan_fonts") + ["TTNorms-Black.otf"]), ),
            "text": ("STRING", {"default": "Text"}),
            "direction": (
            [   'up',
                'down',
                'left',
                'right',
                'overlay'
            ],
            {
            "default": 'up'
             }),
            },
            "optional":{
                "caption": ("STRING", {"default": "", "forceInput": True}),
            }
            }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "addlabel"
    CATEGORY = "Swwan/text"
    DESCRIPTION = """
Creates a new with the given text, and concatenates it to
either above or below the input image.
Note that this changes the input image's height!
Fonts are loaded from this folder:
ComfyUI/custom_nodes/ComfyUI-KJNodes/fonts
"""

    def addlabel(self, image, text_x, text_y, text, height, font_size, font_color, label_color, font, direction, caption=""):
        batch_size = image.shape[0]
        width = image.shape[2]

        font_path = folder_paths.get_full_path("swwan_fonts", "FreeMono.ttf") if font == "TTNorms-Black.otf" else folder_paths.get_full_path("swwan_fonts", "FreeMono.ttf" if font == "TTNorms-Black.otf" else font)

        def process_image(input_image, caption_text):
            font = ImageFont.truetype(font_path, font_size)
            words = caption_text.split()
            lines = []
            current_line = []
            current_line_width = 0

            for word in words:
                word_width = font.getbbox(word)[2]
                if current_line_width + word_width <= width - 2 * text_x:
                    current_line.append(word)
                    current_line_width += word_width + font.getbbox(" ")[2]  # Add space width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_line_width = word_width

            if current_line:
                lines.append(" ".join(current_line))

            if direction == 'overlay':
                pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
            else:
                if height == -1:
                    # Adjust the image height automatically
                    margin = 8
                    required_height = (text_y + len(lines) * font_size) + margin # Calculate required height
                    pil_image = Image.new("RGB", (width, required_height), label_color)
                else:
                    # Initialize with a minimal height
                    label_image = Image.new("RGB", (width, height), label_color)
                    pil_image = label_image

            draw = ImageDraw.Draw(pil_image)


            y_offset = text_y
            for line in lines:
                try:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color, features=['-liga'])
                except:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color)
                y_offset += font_size

            processed_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
            return processed_image

        if caption == "":
            processed_images = [process_image(img, text) for img in image]
        else:
            assert len(caption) == batch_size, f"Number of captions {(len(caption))} does not match number of images"
            processed_images = [process_image(img, cap) for img, cap in zip(image, caption)]
        processed_batch = torch.cat(processed_images, dim=0)

        # Combine images based on direction
        if direction == 'down':
            combined_images = torch.cat((image, processed_batch), dim=1)
        elif direction == 'up':
            combined_images = torch.cat((processed_batch, image), dim=1)
        elif direction == 'left':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((processed_batch, image), dim=2)
        elif direction == 'right':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((image, processed_batch), dim=2)
        else:
            combined_images = processed_batch

        return (combined_images,)

class ImageBatchTestPattern:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "batch_size": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
            "start_from": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
            "text_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "font": ((folder_paths.get_filename_list("swwan_fonts") + ["TTNorms-Black.otf"]), ),
            "font_size": ("INT", {"default": 255,"min": 8, "max": 4096, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generatetestpattern"
    CATEGORY = "Swwan/text"

    def generatetestpattern(self, batch_size, font, font_size, start_from, width, height, text_x, text_y):
        out = []
        # Generate the sequential numbers for each image
        numbers = np.arange(start_from, start_from + batch_size)
        font_path = folder_paths.get_full_path("swwan_fonts", "FreeMono.ttf" if font == "TTNorms-Black.otf" else font)

        for number in numbers:
            # Create a black image with the number as a random color text
            image = Image.new("RGB", (width, height), color='black')
            draw = ImageDraw.Draw(image)

            # Generate a random color for the text
            font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            font = ImageFont.truetype(font_path, font_size)

            # Get the size of the text and position it in the center
            text = str(number)

            try:
                draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
            except:
                draw.text((text_x, text_y), text, font=font, fill=font_color,)

            # Convert the image to a numpy array and normalize the pixel values
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            out.append(image_tensor)
        out_tensor = torch.cat(out, dim=0)

        return (out_tensor,)

class ImageAndMaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_color": ("STRING", {"default": "255, 255, 255"}),
                "pass_through": ("BOOLEAN", {"default": False}),
             },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Preview an image or a mask, when both inputs are used
composites the mask on top of the image.
with pass_through on the preview is disabled and the
composite is returned from the composite slot instead,
this allows for the preview to be passed for video combine
nodes for example.
"""

    def execute(self, mask_opacity, mask_color, pass_through, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            if ',' in mask_color:
                color_list = np.clip([int(channel) for channel in mask_color.split(',')], 0, 255) # RGB format
            else:
                mask_color = mask_color.lstrip('#')
                color_list = [int(mask_color[i:i+2], 16) for i in (0, 2, 4)] # Hex format
            mask_image[:, :, :, 0] = color_list[0] / 255 # Red channel
            mask_image[:, :, :, 1] = color_list[1] / 255 # Green channel
            mask_image[:, :, :, 2] = color_list[2] / 255 # Blue channel

            destination, source = node_helpers.image_alpha_fix(image, mask_image)
            destination = destination.clone().movedim(-1, 1)
            preview = composite(destination, source.movedim(-1, 1), 0, 0, mask_adjusted, 1, True).movedim(1, -1)

        if pass_through:
            return (preview, )
        return(self.save_images(preview, filename_prefix, prompt, extra_pnginfo))

class PreviewAnimation:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     },
                "optional": {
                    "images": ("IMAGE", ),
                    "masks": ("MASK", ),
                },
            }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Swwan/image"

    def preview(self, fps, images=None, masks=None):
        filename_prefix = "AnimPreview"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        pil_images = []

        if images is not None and masks is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)
            for mask in masks:
                if pil_images:
                    mask_np = mask.cpu().numpy()
                    mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)  # Convert to values between 0 and 255
                    mask_img = Image.fromarray(mask_np, mode='L')
                    img = pil_images.pop(0)  # Remove and get the first image
                    img = img.convert("RGBA")  # Convert base image to RGBA

                    # Create a new RGBA image based on the grayscale mask
                    rgba_mask_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    rgba_mask_img.putalpha(mask_img)  # Use the mask image as the alpha channel

                    # Composite the RGBA mask onto the base image
                    composited_img = Image.alpha_composite(img, rgba_mask_img)
                    pil_images.append(composited_img)  # Add the composited image back

        elif images is not None and masks is None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

        elif masks is not None and images is None:
            for mask in masks:
                mask_np = 255. * mask.cpu().numpy()
                mask_img = Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8))
                pil_images.append(mask_img)
        else:
            print("PreviewAnimation: No images or masks provided")
            return { "ui": { "images": results, "animated": (None,), "text": "empty" }}

        num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], lossless=False, quality=50, method=0)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,), "text": [f"{num_frames}x{pil_images[0].size[0]}x{pil_images[0].size[1]}"] } }

class LoadAndResizeImage:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f.name for f in pathlib.Path(input_dir).iterdir() if f.is_file()]
        return {"required":
                    {
                    "image": (sorted(files), {"image_upload": True}),
                    "resize": ("BOOLEAN", { "default": False }),
                    "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "repeat": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                    "keep_proportion": ("BOOLEAN", { "default": False }),
                    "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                    "mask_channel": (s._color_channels, {"tooltip": "Channel to use for the mask output"}),
                    "background_color": ("STRING", { "default": "", "tooltip": "Fills the alpha channel with the specified color."}),
                    },
                }

    CATEGORY = "Swwan/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "width", "height","image_path",)
    FUNCTION = "load_image"

    def load_image(self, image, resize, width, height, repeat, keep_proportion, divisible_by, mask_channel, background_color):
        from PIL import ImageColor, Image, ImageOps, ImageSequence
        import numpy as np
        import torch
        image_path = folder_paths.get_annotated_filepath(image)

        import node_helpers
        img = node_helpers.pillow(Image.open, image_path)

        # Process the background_color
        if background_color:
            try:
                # Try to parse as RGB tuple
                bg_color_rgba = tuple(int(x.strip()) for x in background_color.split(','))
            except ValueError:
                # If parsing fails, it might be a hex color or named color
                if background_color.startswith('#') or background_color.lower() in ImageColor.colormap:
                    bg_color_rgba = ImageColor.getrgb(background_color)
                else:
                    raise ValueError(f"Invalid background color: {background_color}")

            bg_color_rgba += (255,)  # Add alpha channel
        else:
            bg_color_rgba = None  # No background color specified

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        W, H = img.size
        if resize:
            if keep_proportion:
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
            else:
                if width == 0:
                    width = W
                if height == 0:
                    height = H

            if divisible_by > 1:
                width = width - (width % divisible_by)
                height = height - (height % divisible_by)
        else:
            width, height = W, H

        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

            if frame.mode == 'I':
                frame = frame.point(lambda i: i * (1 / 255))

            if frame.mode == 'P':
                frame = frame.convert("RGBA")
            elif 'A' in frame.getbands():
                frame = frame.convert("RGBA")

            # Extract alpha channel if it exists
            if 'A' in frame.getbands() and bg_color_rgba:
                alpha_mask = np.array(frame.getchannel('A')).astype(np.float32) / 255.0
                alpha_mask = 1. - torch.from_numpy(alpha_mask)
                bg_image = Image.new("RGBA", frame.size, bg_color_rgba)
                # Composite the frame onto the background
                frame = Image.alpha_composite(bg_image, frame)
            else:
                alpha_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            image = frame.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue
            if resize:
                image = image.resize((width, height), Image.Resampling.BILINEAR)

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            c = mask_channel[0].upper()
            if c in frame.getbands():
                if resize:
                    frame = frame.resize((width, height), Image.Resampling.BILINEAR)
                mask = np.array(frame.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A' and bg_color_rgba:
                    mask = alpha_mask
                elif c == 'A':
                    mask = 1. - mask
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            if repeat > 1:
                output_image = output_image.repeat(repeat, 1, 1, 1)
                output_mask = output_mask.repeat(repeat, 1, 1)

        return (output_image, output_mask, width, height, image_path)


    # @classmethod
    # def IS_CHANGED(s, image, **kwargs):
    #     image_path = folder_paths.get_annotated_filepath(image)
    #     m = hashlib.sha256()
    #     with open(image_path, 'rb') as f:
    #         m.update(f.read())
    #     return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class LoadImagesFromFolderKJ:
    # Dictionary to store folder hashes
    folder_hashes = {}

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        if folder and not os.path.isabs(folder) and args.base_directory:
            folder = os.path.join(args.base_directory, folder)
        if not folder or not os.path.isdir(folder):
            return float("NaN")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']
        include_subfolders = kwargs.get('include_subfolders', False)

        file_data = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(path)
                            file_data.append((path, mtime))
                        except OSError:
                            pass
        else:
            for file in sorted(os.listdir(folder)):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    path = os.path.join(folder, file)
                    try:
                        mtime = os.path.getmtime(path)
                        file_data.append((path, mtime))
                    except OSError:
                        pass

        file_data.sort()

        combined_hash = hashlib.md5()
        combined_hash.update(folder.encode('utf-8'))
        combined_hash.update(str(len(file_data)).encode('utf-8'))

        for path, mtime in file_data:
            combined_hash.update(f"{path}:{mtime}".encode('utf-8'))

        current_hash = combined_hash.hexdigest()

        old_hash = cls.folder_hashes.get(folder)
        cls.folder_hashes[folder] = current_hash

        if old_hash == current_hash:
            return old_hash

        return current_hash

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "height": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "keep_aspect_ratio": (["crop", "pad", "stretch",],),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "count", "image_path",)
    FUNCTION = "load_images"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

    def load_images(self, folder, width, height, image_load_cap, start_index, keep_aspect_ratio, include_subfolders=False):
        if folder and not os.path.isabs(folder) and args.base_directory:
            folder = os.path.join(args.base_directory, folder)
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']
        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in sorted(os.listdir(folder)):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder, file))

        dir_files = sorted(image_paths)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        pbar = ProgressBar(len(dir_files))

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)

            # Resize image to maximum dimensions
            if width == -1 and height == -1:
                width = i.size[0]
                height = i.size[1]
            if i.size != (width, height):
                i = self.resize_with_aspect_ratio(i, width, height, keep_aspect_ratio)


            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                if mask.shape != (height, width):
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                                         size=(height, width),
                                                         mode='bilinear',
                                                         align_corners=False).squeeze()
            else:
                mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1
            pbar.update(1)

        if len(images) == 1:
            return (images[0], masks[0], 1, image_path_list)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = masks[0].unsqueeze(0)

            for image2 in images[1:]:
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                mask1 = torch.cat((mask1, mask2.unsqueeze(0)), dim=0)

            return (image1, mask1, len(images), image_path_list)
    def resize_with_aspect_ratio(self, img, width, height, mode):
        if mode == "stretch":
            return img.resize((width, height), Image.Resampling.LANCZOS)

        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        target_ratio = width / height

        if mode == "crop":
            # Calculate dimensions for center crop
            if aspect_ratio > target_ratio:
                # Image is wider - crop width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                left = (new_width - width) // 2
                return img.crop((left, 0, left + width, height))
            else:
                # Image is taller - crop height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - height) // 2
                return img.crop((0, top, width, top + height))

        elif mode == "pad":
            pad_color = self.get_edge_color(img)
            # Calculate dimensions for padding
            if aspect_ratio > target_ratio:
                # Image is wider - pad height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                padding = (height - new_height) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (0, padding))
                return padded
            else:
                # Image is taller - pad width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                padding = (width - new_width) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (padding, 0))
                return padded
    def get_edge_color(self, img):
        from PIL import ImageStat
        """Sample edges and return dominant color"""
        width, height = img.size
        img = img.convert('RGBA')

        # Create 1-pixel high/wide images from edges
        top = img.crop((0, 0, width, 1))
        bottom = img.crop((0, height-1, width, height))
        left = img.crop((0, 0, 1, height))
        right = img.crop((width-1, 0, width, height))

        # Combine edges into single image
        edges = Image.new('RGBA', (width*2 + height*2, 1))
        edges.paste(top, (0, 0))
        edges.paste(bottom, (width, 0))
        edges.paste(left.resize((height, 1)), (width*2, 0))
        edges.paste(right.resize((height, 1)), (width*2 + height, 0))

        # Get median color
        stat = ImageStat.Stat(edges)
        median = tuple(map(int, stat.median))
        return median

class SaveImageKJ:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to."}),
            },
            "optional": {
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Swwan/image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, output_folder, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, caption=None, caption_file_extension=".txt"):
        filename_prefix += self.prefix_append

        if os.path.isabs(output_folder):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder
            _, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_folder, images[0].shape[1], images[0].shape[0])
        else:
            base_dir = folder_paths.get_output_directory()
            self.output_dir = _legacy_output_folder(output_folder, base_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = _build_png_metadata(prompt, extra_pnginfo)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            base_file_name = f"{filename_with_batch_num}_{counter:05}_"
            file = f"{base_file_name}.png"
            _save_image_with_fallback(img, os.path.join(full_output_folder, file), {"format":"PNG", "pnginfo":metadata, "compress_level":self.compress_level})
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            if caption is not None:
                txt_file = base_file_name + caption_file_extension
                file_path = os.path.join(full_output_folder, txt_file)
                with open(file_path, 'w') as f:
                    f.write(caption)

            counter += 1

        return file,

class SaveStringKJ:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}),
                "filename_prefix": ("STRING", {"default": "text", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to."}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_string"

    OUTPUT_NODE = True

    CATEGORY = "Swwan/misc"
    DESCRIPTION = "Saves the input string to your ComfyUI output directory."

    def save_string(self, string, output_folder, filename_prefix="text", file_extension=".txt"):
        filename_prefix += self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        if output_folder and not os.path.isabs(output_folder) and args.base_directory:
            output_folder = os.path.join(args.base_directory, output_folder)
        if output_folder != "output":
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder

        base_file_name = f"{filename_prefix}_{counter:05}_"
        results = list()

        txt_file = base_file_name + file_extension
        file_path = os.path.join(full_output_folder, txt_file)
        with open(file_path, 'w') as f:
            f.write(string)

        return results,

class FastPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "quality" : ("INT", {"default": 75, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "Swwan/experimental"
    OUTPUT_NODE = True
    DESCRIPTION = "Experimental node for faster image previews by displaying through base64 it without saving to disk."

    def preview(self, image, format, quality):
        from torchvision.transforms.functional import to_pil_image
        pil_image = to_pil_image(image[0].permute(2, 0, 1))

        with BytesIO() as buffered:
            pil_image.save(buffered, format=format, quality=quality)
            img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return {
            "ui": {"bg_image": [img_base64]},
            "result": ()
        }

class LoadVideosFromFolder:
    @classmethod
    def __init__(cls):
        try:
            cls.vhs_nodes = importlib.import_module("ComfyUI-VideoHelperSuite.videohelpersuite")
        except ImportError:
            try:
                cls.vhs_nodes = importlib.import_module("comfyui-videohelpersuite.videohelpersuite")
            except ImportError:
                # Fallback to sys.modules search for Windows compatibility
                import sys
                vhs_module = None
                for module_name in sys.modules:
                    if 'videohelpersuite' in module_name and 'videohelpersuite' in sys.modules[module_name].__dict__:
                        vhs_module = sys.modules[module_name]
                        break

                if vhs_module is None:
                    # Try direct access to the videohelpersuite submodule
                    for module_name in sys.modules:
                        if module_name.endswith('videohelpersuite'):
                            vhs_module = sys.modules[module_name]
                            break

                if vhs_module is not None:
                    cls.vhs_nodes = vhs_module
                else:
                    raise ImportError("This node requires ComfyUI-VideoHelperSuite to be installed.")

        except ImportError:
            raise ImportError("This node requires ComfyUI-VideoHelperSuite to be installed.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "X://insert/path/"},),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, 'disable': 0}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "disable": 0}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "output_type": (["batch", "grid"], {"default": "batch"}),
                "grid_max_columns": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "disable": 1}),
                "add_label": ( "BOOLEAN", {"default": False} ),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Swwan/misc"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )

    FUNCTION = "load_video"

    def load_video(self, output_type, grid_max_columns, add_label=False, **kwargs):
        if kwargs.get('video') and not os.path.isabs(kwargs['video']) and args.base_directory:
            kwargs['video'] = os.path.join(args.base_directory, kwargs['video'])

        if self.vhs_nodes is None:
            raise ImportError("This node requires ComfyUI-VideoHelperSuite to be installed.")
        videos_list = []
        filenames = []
        for f in sorted(os.listdir(kwargs['video'])):
            if os.path.isfile(os.path.join(kwargs['video'], f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in ['webm', 'mp4', 'mkv', 'gif', 'mov']):
                    videos_list.append(os.path.join(kwargs['video'], f))
                    filenames.append(f)
        print(videos_list)
        kwargs.pop('video')
        loaded_videos = []
        for idx, video in enumerate(videos_list):
            video_tensor = self.vhs_nodes.load_video_nodes.load_video(video=video, **kwargs)[0]
            if add_label:
                # Add filename label above video (without extension)
                if video_tensor.dim() == 4:
                    _, h, w, c = video_tensor.shape
                else:
                    h, w, c = video_tensor.shape
                # Remove extension from filename
                label_text = filenames[idx].rsplit('.', 1)[0]
                font_size = max(16, w // 20)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                dummy_img = Image.new("RGB", (w, 10), (0,0,0))
                draw = ImageDraw.Draw(dummy_img)
                text_bbox = draw.textbbox((0,0), label_text, font=font)
                extra_padding = max(12, font_size // 2)  # More padding under the font
                label_height = text_bbox[3] - text_bbox[1] + extra_padding
                label_img = Image.new("RGB", (w, label_height), (0,0,0))
                draw = ImageDraw.Draw(label_img)
                draw.text((w//2 - (text_bbox[2]-text_bbox[0])//2, 4), label_text, font=font, fill=(255,255,255))
                label_np = np.asarray(label_img).astype(np.float32) / 255.0
                label_tensor = torch.from_numpy(label_np)
                if c == 1:
                    label_tensor = label_tensor.mean(dim=2, keepdim=True)
                elif c == 4:
                    alpha = torch.ones((label_height, w, 1), dtype=label_tensor.dtype)
                    label_tensor = torch.cat([label_tensor, alpha], dim=2)
                if video_tensor.dim() == 4:
                    label_tensor = label_tensor.unsqueeze(0).expand(video_tensor.shape[0], -1, -1, -1)
                    video_tensor = torch.cat([label_tensor, video_tensor], dim=1)
                else:
                    video_tensor = torch.cat([label_tensor, video_tensor], dim=0)
            loaded_videos.append(video_tensor)
        if output_type == "batch":
            out_tensor = torch.cat(loaded_videos)
        elif output_type == "grid":
            rows = (len(loaded_videos) + grid_max_columns - 1) // grid_max_columns
            # Pad the last row if needed
            total_slots = rows * grid_max_columns
            while len(loaded_videos) < total_slots:
                loaded_videos.append(torch.zeros_like(loaded_videos[0]))
            # Create grid by rows
            row_tensors = []
            for row_idx in range(rows):
                start_idx = row_idx * grid_max_columns
                end_idx = start_idx + grid_max_columns
                row_videos = loaded_videos[start_idx:end_idx]
                # Pad all videos in this row to the same height
                heights = [v.shape[1] for v in row_videos]
                max_height = max(heights)
                padded_row_videos = []
                for v in row_videos:
                    pad_height = max_height - v.shape[1]
                    if pad_height > 0:
                        # Pad (frames, H, W, C) or (H, W, C)
                        if v.dim() == 4:
                            pad = (0,0, 0,0, 0,pad_height, 0,0)  # (C,W,H,F)
                            v = torch.nn.functional.pad(v, (0,0,0,0,0,pad_height,0,0))
                        else:
                            v = torch.nn.functional.pad(v, (0,0,0,0,pad_height,0))
                    padded_row_videos.append(v)
                row_tensor = torch.cat(padded_row_videos, dim=2)  # Concatenate horizontally
                row_tensors.append(row_tensor)
            out_tensor = torch.cat(row_tensors, dim=1)  # Concatenate rows vertically
        print(out_tensor.shape)
        return out_tensor,

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        if s.vhs_nodes is not None:
            return s.vhs_nodes.utils.hash_path(video)
        return None
