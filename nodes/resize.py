# SPDX-License-Identifier: GPL-3.0-only
# Derived image algorithms: ComfyUI-KJNodes.
from ..ops.image_common import F, MAX_RESOLUTION, PromptServer, common_upscale, math, model_management, os, time, torch
from .mask import ImagePadKJ

class ImageResizeKJ:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                #"width_input": ("INT", { "forceInput": True}),
                #"height_input": ("INT", { "forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["disabled","center", 0], { "tooltip": "0 will do the default center crop, this is a workaround for the widget order changing with the new frontend, as in old workflows the value of this widget becomes 0 automatically" }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Swwan/image"
    DEPRECATED = True
    DESCRIPTION = """
DEPRECATED!

Due to ComfyUI frontend changes, this node should no longer be used, please check the
v2 of the node. This node is only kept to not completely break older workflows.

"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by,
               width_input=None, height_input=None, get_image_size=None, crop="disabled"):
        B, H, W, C = image.shape

        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape

        if keep_proportion and get_image_size is None:
                # If one of the dimensions is zero, calculate it to maintain the aspect ratio
                if width == 0 and height != 0:
                    ratio = height / H
                    width = round(W * ratio)
                elif height == 0 and width != 0:
                    ratio = width / W
                    height = round(H * ratio)
                elif width != 0 and height != 0:
                    # Scale based on which dimension is smaller in proportion to the desired dimensions
                    ratio = min(width / W, height / H)
                    width = round(W * ratio)
                    height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        if crop == 0: #workaround for old workflows
            crop = "center"

        image = image.movedim(-1,1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image, image.shape[2], image.shape[1],)

class ImageResizeKJv2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"], { "default": "stretch" }),
                "pad_color": ("STRING", { "default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
                "resize_mode": (["standard", "edit_size", "aspect_ratio", "essentials"], {"default": "standard"}),
                "size_rule": (["按长边等比例", "按短边等比例", "自定义宽高"], {"default": "按长边等比例"}),
                "edge_length": ("INT", {"default": 1024, "min": 64, "max": 100000}),
                "execute_condition": (["总是", "最长边大于时", "最小边小于时"], {"default": "总是"}),
                "edit_fit": (["拉伸", "裁剪", "填充_自定颜色", "填充_边框颜色", "填充_边缘像素", "总像素_等比例"], {"default": "裁剪"}),
                "fill_color": ("COLORCODE", {"default": "#364254"}),
                "aspect_ratio": (["original","custom","1:1","3:2","4:3","16:9","2:3","3:4","9:16"], {"default":"original"}),
                "proportional_width": ("INT", {"default":1,"min":1}),
                "proportional_height": ("INT", {"default":1,"min":1}),
                "aspect_fit": (["letterbox","crop","fill"], {"default":"letterbox"}),
                "aspect_method": (["lanczos","bicubic","hamming","bilinear","box","nearest"], {"default":"lanczos"}),
                "aspect_round": (["8","16","32","64","128","256","512","None"], {"default":"8"}),
                "aspect_scale_side": (["None","longest","shortest","width","height","total_pixel(kilo pixel)"], {"default":"longest"}),
                "aspect_length": ("INT", {"default":1024,"min":4}),
                "essentials_method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "essentials_condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "essentials_interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                #"per_batch": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Process images in sub-batches to reduce memory usage. 0 disables sub-batching."}),
            },
             "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Resizes the image to the specified width and height.
Size can be retrieved from the input.

Keep proportions keeps the aspect ratio of the image, by
highest dimension.
"""

    DESCRIPTION += "\nEdit size: CPU reference sizing, fit and conditional execution; fill_color is COLORCODE. Standard keeps the existing GPU/RTX algorithm.\n"

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, unique_id, device="cpu", mask=None, per_batch=64,
               resize_mode="standard", size_rule="按长边等比例", edge_length=1024,
               execute_condition="总是", edit_fit="裁剪", fill_color="#364254",
               aspect_ratio="original", proportional_width=1, proportional_height=1,
               aspect_fit="letterbox", aspect_method="lanczos", aspect_round="8", aspect_scale_side="longest", aspect_length=1024, essentials_method="stretch", essentials_condition="always", essentials_interpolation="nearest"):
        if resize_mode == "essentials":
            from ..ops.resize_essentials import EssentialsResizeAlgorithm
            algorithm = EssentialsResizeAlgorithm()
            resized, w, h = algorithm.execute(image, width, height, essentials_method, essentials_interpolation, essentials_condition, divisible_by)
            resized_mask = None
            if mask is not None:
                resized_mask = algorithm.execute(mask.unsqueeze(-1), width, height, essentials_method, essentials_interpolation, essentials_condition, divisible_by)[0][..., 0]
            return resized, w, h, resized_mask
        if resize_mode == "aspect_ratio":
            from ..ops.aspect_resize import aspect_resize
            result = aspect_resize(
                aspect_ratio, proportional_width, proportional_height, aspect_fit, aspect_method,
                aspect_round, aspect_scale_side, aspect_length, fill_color, image, mask)
            return (result[0], result[3], result[4], result[1])
        if resize_mode == "edit_size":
            from ..edit_image_ops import EditImageResize

            interpolation = {
                "bilinear": "双线性插值", "bicubic": "双三次插值",
                "area": "区域", "nearest-exact": "邻近-精确", "lanczos": "Lanczos",
            }
            if upscale_method not in interpolation:
                raise ValueError("Edit size supports bilinear, bicubic, area, nearest-exact and lanczos")
            positions = {"center": "居中", "top": "上", "bottom": "下", "left": "左", "right": "右"}
            return EditImageResize().执行缩放(
                image, size_rule, width, height, edge_length,
                interpolation[upscale_method], edit_fit, positions[crop_position],
                execute_condition, fill_color, divisible_by, mask,
            )
        B, H, W, C = image.shape

        is_rtx_vsr = upscale_method == "nvidia_rtx_vsr"
        if is_rtx_vsr:
            if not torch.cuda.is_available():
                raise RuntimeError("NVIDIA RTX Video Super Resolution requires a CUDA-capable NVIDIA GPU.")
            device = model_management.get_torch_device()
            if getattr(device, "type", None) != "cuda":
                raise RuntimeError("NVIDIA RTX Video Super Resolution requires ComfyUI to use a CUDA device.")
        elif device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"

        # Initialize padding variables
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion in ["resize", "total_pixels"] or keep_proportion.startswith("pad") or pillarbox_blur:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = W / H
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))

            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            elif width == 0 and height == 0:
                new_width = W
                new_height = H
            elif width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
                new_height = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_width = width
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)
            else:
                new_width = width
                new_height = height

            if keep_proportion.startswith("pad") or pillarbox_blur:
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        if is_rtx_vsr:
            width = max(8, round(width / 8) * 8)
            height = max(8, round(height / 8) * 8)

        # Preflight estimate (log-only when batching is active)
        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()  # typically 4 for float32
                est_total_bytes = B * height * width * C * bytes_per_elem
                est_mb = est_total_bytes / (1024 * 1024)
                msg = f"<tr><td>Resize v2</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>"
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(msg, unique_id)
                    except:
                        pass
                else:
                    print(f"[ImageResizeKJv2] estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}")
            except:
                pass

        nvvfx_sr = None
        nvvfx_ctx = None
        if is_rtx_vsr:
            try:
                import nvvfx
            except ImportError as error:
                raise ImportError(
                    "NVIDIA RTX Video Super Resolution is not available. "
                    "Install the optional nvidia-vfx/nvvfx package and use a compatible NVIDIA GPU."
                ) from error

            try:
                nvvfx_ctx = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
                nvvfx_sr = nvvfx_ctx.__enter__()
                nvvfx_sr.output_width = width
                nvvfx_sr.output_height = height
                nvvfx_sr.load()
            except Exception:
                if nvvfx_ctx is not None:
                    nvvfx_ctx.__exit__(None, None, None)
                raise

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            # Avoid unnecessary clones; only move if needed
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            # Crop logic
            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                if old_aspect > new_aspect:
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2
                out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

            if is_rtx_vsr:
                frames_chw = out_image.movedim(-1, 1).to(device).contiguous()
                upscaled_frames = []
                for frame in frames_chw:
                    dlpack_out = nvvfx_sr.run(frame).image
                    upscaled_frames.append(torch.from_dlpack(dlpack_out).clone())
                out_image = torch.stack(upscaled_frames, dim=0).movedim(1, -1).cpu()
                if out_mask is not None:
                    out_mask = common_upscale(out_mask.unsqueeze(1), width, height, "bilinear", crop="disabled").squeeze(1)
            else:
                out_image = common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)
                if out_mask is not None:
                    if upscale_method == "lanczos":
                        out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
                    else:
                        out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)

            # Pad logic
            if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height

                pad_mode = (
                    "pillarbox_blur" if pillarbox_blur else
                    "edge" if keep_proportion == "pad_edge" else
                    "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                    "color"
                )
                out_image, out_mask = ImagePadKJ.pad(self, out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode, mask=out_mask)

            return out_image, out_mask

        try:
            # If batching disabled (per_batch==0) or batch fits, process whole batch
            if per_batch == 0 or B <= per_batch:
                out_image, out_mask = _process_subbatch(image, mask, pad_left, pad_right, pad_top, pad_bottom)
            else:
                chunks = []
                mask_chunks = [] if mask is not None else None
                total_batches = (B + per_batch - 1) // per_batch
                current_batch = 0
                for start_idx in range(0, B, per_batch):
                    current_batch += 1
                    end_idx = min(start_idx + per_batch, B)
                    sub_img = image[start_idx:end_idx]
                    sub_mask = mask[start_idx:end_idx] if mask is not None else None
                    sub_out_img, sub_out_mask = _process_subbatch(sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                    chunks.append(sub_out_img.cpu())
                    if mask is not None:
                        mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                    # Per-batch progress update
                    if unique_id and PromptServer is not None:
                        try:
                            PromptServer.instance.send_progress_text(
                                f"<tr><td>Resize v2</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}</td></tr>",
                                unique_id
                            )
                        except:
                            pass
                    else:
                        try:
                            print(f"[ImageResizeKJv2] batch {current_batch}/{total_batches} · images {end_idx}/{B}")
                        except:
                            pass
                out_image = torch.cat(chunks, dim=0)
                if mask is not None and any(m is not None for m in mask_chunks):
                    out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
                else:
                    out_mask = None
        finally:
            if nvvfx_ctx is not None:
                nvvfx_ctx.__exit__(None, None, None)

        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if out_mask is not None else torch.zeros(64,64, device=torch.device("cpu"), dtype=torch.float32))

class ImageResizeByMegapixels:
    """
    Resize image by target megapixels with aspect ratio control.
    Calculates optimal dimensions based on target total pixels and divisibility requirements.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
    aspect_ratios = ["default", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]
    divisible_options = [2, 4, 8, 16, 32, 64]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Target megapixels (1.0 = 1 million pixels). Use 0 to skip resize and keep the original size."}),
                "aspect_ratio": (s.aspect_ratios, {"default": "default", "tooltip": "Target aspect ratio. 'default' keeps original ratio."}),
                "keep_proportion": (["crop", "resize", "pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur"], {"default": "crop", "tooltip": "How to handle aspect ratio change when not using 'default'."}),
                "divisible_by": (s.divisible_options, {"default": 16, "tooltip": "Width and height will be divisible by this value."}),
                "default_divisible": ("BOOLEAN", {"default": False, "tooltip": "When enabled, ensures final output dimensions strictly follow divisible_by constraint."}),
                "upscale_method": (s.upscale_methods, {"default": "bilinear"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color to use for padding."}),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask", "longest_edge",)
    FUNCTION = "resize"
    CATEGORY = "Swwan/image"
    DESCRIPTION = """
Resizes image to target megapixels with optional aspect ratio control.

- megapixels: Target total pixels in millions (1.0 = 1,000,000 pixels). Use 0 to skip resizing.
- aspect_ratio: 'default' keeps original ratio, or choose a specific ratio
- keep_proportion: How to handle aspect ratio changes (crop, pad, resize, etc.)
- divisible_by: Ensures output dimensions are divisible by this value
- default_divisible: When enabled, strictly enforces divisible_by on final output
"""

    @staticmethod
    def parse_aspect_ratio(ratio_str):
        """Parse aspect ratio string like '16:9' to (16, 9)"""
        if ratio_str == "default":
            return None
        parts = ratio_str.split(":")
        return (int(parts[0]), int(parts[1]))

    def resize(self, image, megapixels, aspect_ratio, keep_proportion, divisible_by, default_divisible,
               upscale_method, crop_position, pad_color, unique_id=None, device="cpu", mask=None, per_batch=64):
        B, H, W, C = image.shape
        empty_mask = torch.zeros(64, 64, device=torch.device("cpu"), dtype=torch.float32)

        if megapixels <= 0:
            return (
                image,
                W,
                H,
                mask if mask is not None else empty_mask,
                max(H, W),
            )

        target_pixels = megapixels * 1_000_000

        # Determine aspect ratio
        if aspect_ratio == "default":
            ratio_w, ratio_h = W, H
        else:
            ratio_w, ratio_h = self.parse_aspect_ratio(aspect_ratio)

        # Calculate scale factor and new dimensions
        scale = math.sqrt(target_pixels / (ratio_w * ratio_h))
        new_width = math.floor(ratio_w * scale / divisible_by) * divisible_by
        new_height = math.floor(ratio_h * scale / divisible_by) * divisible_by

        # Ensure minimum size
        if new_width < divisible_by:
            new_width = divisible_by
        if new_height < divisible_by:
            new_height = divisible_by

        if (
            aspect_ratio == "default"
            and mask is None
            and device == "cpu"
            and upscale_method not in ("lanczos", "nvidia_rtx_vsr")
            and (per_batch == 0 or B <= per_batch)
        ):
            resize_start = time.perf_counter() if os.environ.get("SWWAN_RESIZE_DEBUG") == "1" else None

            # Match ImageResizeKJv2's "resize" mode dimensions without the generic crop/pad/mask path.
            ratio = min(new_width / W, new_height / H)
            final_width = round(W * ratio)
            final_height = round(H * ratio)
            if default_divisible and divisible_by > 1:
                final_width = final_width - (final_width % divisible_by)
                final_height = final_height - (final_height % divisible_by)

            source_image = image if image.device.type == "cpu" else image.to(torch.device("cpu"))
            out_image = F.interpolate(
                source_image.movedim(-1, 1),
                size=(final_height, final_width),
                mode=upscale_method,
            ).movedim(1, -1)

            if resize_start is not None:
                elapsed = time.perf_counter() - resize_start
                print(
                    f"[ImageResizeByMegapixels] fast cpu resize "
                    f"{B}x{W}x{H} -> {final_width}x{final_height} "
                    f"({upscale_method}) in {elapsed:.4f}s"
                )

            return (
                out_image,
                out_image.shape[2],
                out_image.shape[1],
                empty_mask,
                max(out_image.shape[1], out_image.shape[2]),
            )

        # Use ImageResizeKJv2's resize logic
        resizer = ImageResizeKJv2()

        # When aspect_ratio is "default", we can use "resize" mode directly
        # When aspect_ratio is different, we need to use the keep_proportion to handle it
        if aspect_ratio == "default":
            # Original aspect ratio is maintained, just resize
            mode = "resize"
        else:
            # Aspect ratio change, use the specified keep_proportion mode
            mode = keep_proportion

        out_image, out_width, out_height, out_mask = resizer.resize(
            image=image,
            width=new_width,
            height=new_height,
            keep_proportion=mode,
            upscale_method=upscale_method,
            divisible_by=divisible_by if default_divisible else 1,
            pad_color=pad_color,
            crop_position=crop_position,
            unique_id=unique_id,
            device=device,
            mask=mask,
            per_batch=per_batch
        )
        return out_image, out_width, out_height, out_mask, max(out_height, out_width)
