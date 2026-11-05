# SPDX-License-Identifier: MIT
# Source: LayerStyle; original notices and modification history in THIRD_PARTY_NOTICES.md.
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy import model_management

from .layerstyle_utils import log, tensor2pil, pil2tensor, mask2image, image2mask
from .layerstyle_utils import min_bounding_rect, max_inscribed_rect, mask_area
from .layerstyle_utils import num_round_up_to_multiple, draw_rect


class CropByMaskV5:
    """传统批次裁剪、单张编辑裁剪和 Qwen 对齐面积预算；模式合同见界面帮助。"""

    def __init__(self):
        self.NODE_NAME = 'CropByMask V5'

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        batch_mode_list = ['single_frame', 'batch_first_reuse']
        reserve_mode_list = ['absolute', 'ratio']
        inputs = {
            "required": {
                "detect": (detect_mode,),
                "reserve_mode": (reserve_mode_list,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "top_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "bottom_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "left_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "right_reserve_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "reserve_max": ("INT", {"default": 100, "min": 0, "max": 1024, "step": 1}),
                "round_to_multiple": (multiple_list,),
                "batch_mode": (batch_mode_list,),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "crop_box": ("BOX",),
                "crop_mode": (["bounds", "edit_region", "qwen21_aligned"], {"default": "bounds"}),
                "mask": ("MASK",),
                "fill_mask_holes": ("BOOLEAN", {"default": False}),
                "output_size": (["原像素", "原像素1：1", "自定义宽高"], {"default": "原像素"}),
                "custom_width": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "custom_height": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "alignment": ("INT", {"default": 8, "min": 0, "max": 256}),
                "edit_expansion_mode": (["reserve_ratio", "factor"], {"default": "reserve_ratio"}),
                **{name: ("FLOAT", {"default": 1.2, "min": 1.0, "max": 10.0, "step": 0.1}) for name in
                   ("edit_top_factor", "edit_bottom_factor", "edit_left_factor", "edit_right_factor")},
                "qwen_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "qwen_tolerance": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.1}),
                "qwen_restore_backend": (["swwan", "goohai"], {"default": "swwan"}),
                "qwen_resolution_mode": (["fixed", "crop_area"], {"default": "fixed"}),
                "qwen_max_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 1}),
                "qwen_min_resolution": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 1}),
            }
        }
        tips = {
            "image": "待裁剪的干净原图。edit_region/qwen21_aligned 必须连接，且仅处理首张；mask 必须与该图同尺寸、同坐标。",
            "mask": "仅 edit_region/qwen21_aligned：白色为选区，黑色为非选区，优先于 mask_image。LoadImage 的 MASK 是 1-alpha，不是自动主体选区。aligned 按 >0 取外接框，保留软值；低于512预算缩小时按面积覆盖采样，细小选区可能变灰但不直接丢弃。空选区报错。",
            "mask_image": "遮罩的 IMAGE 输入。bounds 使用灰度遮罩；编辑模式未接 mask 时才使用它，edit_region 转灰度，aligned 读取红通道。请与原图保持同尺寸。",
            "crop_box": "仅 bounds：复用 [left, top, right, bottom] 原图坐标框，右/下为排他边界；连接后跳过自动检测和边距计算。不是 BBOX/SEAM。",
            "crop_mode": "bounds：传统边界裁剪/批次复用；edit_region：局部编辑裁剪与恢复记录；qwen21_aligned：局部裁剪 + BlueBox 32 对齐/不动点校验 + Qwen 面积预算。模式决定其他控件是否生效。",
            "detect": "仅 bounds 且未接 crop_box：mask_area 按遮罩区域，min_bounding_rect 最小外接矩形，max_inscribed_rect 最大内接矩形；后两者使用平滑后的遮罩。",
            "reserve_mode": "仅 bounds：absolute 使用四向像素边距；ratio 上下按遮罩框高度、左右按宽度乘比例，并受 reserve_max 限制。编辑模式改用 edit_expansion_mode。",
            "reserve_max": "仅 bounds + ratio：每侧边距的最大像素数。对 edit_region/qwen21_aligned 的上下文扩展不生效。",
            "round_to_multiple": "仅 bounds：裁剪框宽高向上对齐指定倍数，尽量限制在原图内。不是 aligned 的 Qwen 32 对齐，也不是遮罩块化。",
            "batch_mode": "仅 bounds：两个选项均按首帧遮罩/同一 crop_box 裁剪输入批次，不逐帧检测；batch_first_reuse 显式表示复用。edit_region/aligned 始终只处理首张。",
            "device": "仅 bounds 自动检测时选择平滑计算设备，后续检测仍含 CPU/PIL。不是 Qwen 推理设备；edit_region/aligned 不使用此参数。",
            "fill_mask_holes": "仅编辑模式：填充选区内部封闭孔洞，可能把原本保留的内部区域纳入选区；不是最终保护区。aligned 保留已有软值，仅补孔。默认关闭。",
            "output_size": "仅 edit_region：原像素保持裁剪像素尺度；原像素1：1 扩成正方形；自定义宽高按指定尺寸处理。aligned 由面积预算决定画布，不使用此项。",
            "alignment": "仅 edit_region：原像素/原像素1：1尺寸向上对齐此倍数，自定义输出按该模式规则取整；0/1 不额外对齐。aligned 固定使用自己的 32 倍数候选，不读取此项。",
            "edit_expansion_mode": "仅编辑模式：factor 使用四向 edit_*_factor；reserve_ratio 使用四向 *_reserve_ratio（内部转为 1+ratio）。均按选区外接框短边增加上下文，不是整个宽高乘倍数。",
            "qwen_resolution": "仅 aligned + fixed：固定面积预算 R，目标画布约 R² 像素，不是固定 R×R，也不是长边上限。crop_area 不读取此项。",
            "qwen_tolerance": "仅 aligned：BlueBox 候选面积容差，越大可选画布越多；不是位移容忍像素。1.0 在 R=1024 时约为 3.125% 面积容差；最终仍验证不动点。",
            "qwen_restore_backend": "仅 aligned：swwan 的 region_info 接 SwwanRestoreCropBoxV4；goohai 接原生 GH_CropRestore，保留该节点的校色/羽化能力。本节点只准备几何记录，不执行校色，两个 SEAM 合同不可混接。",
            "qwen_resolution_mode": "仅 aligned：fixed 使用 qwen_resolution；crop_area 按原生选区框+上下文的 sqrt(W×H) 取最近 32 档（半档向上），再限制在 min/max 内。预算在缩放/比例扩框/补边之前计算，不按白色像素总数计算。",
            "qwen_min_resolution": "仅 aligned + crop_area：面积预算下限，默认 128，向上取 32 倍数（129→160）。不是最短边；小裁剪可能被放大到约 min²。低于 512 的模型质量/速度未验收。对齐后的下限不能超过上限。",
            "qwen_max_resolution": "仅 aligned + crop_area：面积预算上限，默认 1024，向下取 32 倍数（1560→1536）。不是最长边；狭长画布单边可以更大。降低上限可减少大裁剪的推理像素，也可能损失细节。",
        }
        for side, label in (("top", "上"), ("bottom", "下"), ("left", "左"), ("right", "右")):
            tips[f"{side}_reserve"] = f"仅 bounds + absolute：向{label}保留的像素边距；负值收缩，边界受原图限制。编辑模式不使用此项。"
            tips[f"{side}_reserve_ratio"] = f"向{label}扩展比例。bounds + ratio 按对应宽/高计算并受 reserve_max 限制；编辑模式 + reserve_ratio 按选区框短边 S×ratio 增加上下文，不受 reserve_max 限制。"
            tips[f"edit_{side}_factor"] = f"仅编辑模式 + factor：向{label}增加 S×(factor-1) 像素，S 为选区框短边；1=不扩展，1.3=增加约 0.3S，不是整张图放大1.3倍。aligned 向上取整，edit_region 截断取整。上下文不自动获得最终修改权限。"
        for axis in ("width", "height"):
            tips[f"custom_{axis}"] = "仅 edit_region + 自定义宽高：与另一维共同设置输出；仅一维为0时，非零值作为长边预算；两维不能同时为0。aligned 不使用此项。"
        for group in inputs.values():
            for name, data in group.items():
                group[name] = (data[0], {**(data[1] if len(data) > 1 else {}), "tooltip": tips[name]})
        return inputs

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOX", "IMAGE", "SEAM", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box", "box_preview", "region_info", "region_mask", "width", "height", "encoder_resolution", "native_crop_width", "native_crop_height", "working_resolution")
    OUTPUT_TOOLTIPS = (
        "实际裁剪/对齐后的图像。aligned 下可接任务标记处理再送编码器；保留干净图作颜色参考。",
        "与裁剪图同尺寸的三通道遮罩预览（IMAGE）；需要 MASK 连线时使用 region_mask。",
        "原图坐标 BOX：[left, top, right, bottom]，右/下排他。aligned 的缩放/补边恢复还必须带同源 region_info，不能只靠 BOX。",
        "原图或遮罩上的裁剪框预览，供检查选区是否完整；不是模型输入。",
        "同源 SEAM，记录原图、裁剪、缩放和补边。aligned 按 qwen_restore_backend 连接匹配恢复器；edit_region 接 SwwanRestoreCropBoxV4；bounds 为 None。",
        "与裁剪图同步变换后的原生 MASK，用于标记或同源贴回；不等于最终硬保护范围或生成 alpha。",
        "实际输出画布宽度；aligned 下为32倍数，可能大于面积预算数值。",
        "实际输出画布高度；aligned 下为32倍数，可能大于面积预算数值。",
        "供官方 TextEncodeQwenImage21.resolution 使用，aligned 输出0以保持已对齐画布；采样使用同一编码器的 latent。其他模式也返回0，但不保证已32对齐。",
        "原生裁剪宽度：aligned 中为选区框+上下文，尚未进行比例扩框、缩放或补边。用于理解预算来源。",
        "原生裁剪高度：aligned 中为选区框+上下文，尚未进行比例扩框、缩放或补边。用于理解预算来源。",
        "实际选中的面积预算R（目标面积约R²，不等于width或height）；aligned fixed/dynamic均返回实际值，其他模式返回0。",
    )
    FUNCTION = 'crop_by_mask_v5'
    CATEGORY = 'Swwan/image'
    DESCRIPTION = """## 局部裁剪 / BlueBox / 面积预算（Swwan）

按选区裁剪、保留上下文并准备恢复坐标。**本节点不执行 Qwen 推理、颜色校正或最终贴回。** 输入参数、输出端口均有中文悬停说明。

### 先选择 crop_mode
- **bounds**：传统边界裁剪。使用 mask_image 或 crop_box；直接 mask 不生效。detect、reserve_mode、reserve_max、round_to_multiple、batch_mode、device 只在此模式使用。输入批次共用首帧选区/同一框，不逐帧检测。
- **edit_region**：单张局部编辑裁剪。使用 image + mask（优先）或 mask_image；支持孔洞填充、上下文、output_size/custom_width/custom_height/alignment，输出 SEAM；空选区沿用旧行为取整图。只处理首张，不是 Qwen BlueBox 对齐模式。
- **qwen21_aligned**：单张 Qwen 局部裁剪。完整 mask 外接框 + 上下文 → 面积预算 → BlueBox 候选评分/32 对齐/不动点校验 → 图与 mask 同步缩放补边。空选区报错；不使用旧的 output_size、alignment、bounds 检测/边距控件。不需要再串接 BlueBox Cut。

### Qwen 动态面积预算
选择 **qwen_resolution_mode=crop_area**，默认 **qwen_min_resolution=128 / qwen_max_resolution=1024**。
W、H 是加上下文后的原生裁剪尺寸，在比例扩框/缩放/补边之前计算：
`R = 32 × floor(sqrt(W × H) / 32 + 0.5)`，再限制到 `[32×ceil(min/32), 32×floor(max/32)]`。
min 向上取整（129→160），max 向下取整（1560→1536）；对齐后 min>max 报错。
**R 是面积预算，目标面积约 R²，不是固定正方形，也不是长边/短边限制。** 实际宽高还受比例与候选容差影响；不是对 mask 白色像素数量求面积。小裁剪会被下限放大，大裁剪被上限缩小；128等低于512的预算尚无模型效果/速度验收。
选择 **fixed** 时只使用 qwen_resolution（默认1024），min/max 不生效。默认 crop_mode=bounds、resolution_mode=fixed 保留，需主动选择 aligned + crop_area 才使用动态预算。

### 上下文与对齐
编辑模式的 edit_expansion_mode=factor 使用四向 edit_*_factor：每侧增加选区框短边 S×(factor−1)，1不扩展，1.3增加约0.3S；reserve_ratio 改用 S×四向ratio。不是整个宽高乘factor，亦不受 bounds 的 reserve_max 限制。
aligned 先利用原图真实上下文，为候选比例扩框，边界不足才复制边缘补图（mask补0），不通过中心裁切丢掉选区。qwen_tolerance 控制候选面积容差，不是位移精度；1.0在R=1024时约3.125%。这些处理减少管线尺寸偏差，**不保证模型生成内容零位移**。

### 遮罩与恢复接线
编辑模式的 mask 必须与原图同尺寸、同坐标；LoadImage 的 MASK 是1-alpha，不是自动主体选区。fill_mask_holes会改变内部孔洞，慎用于保护孔。上下文只是参考，不自动扩大最终允许修改范围。低于512预算缩小遮罩时使用面积覆盖采样，保留细线/离散小选区的非零覆盖；二值遮罩边缘可能变为灰度，不强制阈值化。若缩放后完全失去选区则报错，请提高预算或加宽选区；512及以上保留既有采样。
aligned 的 cropped_image/region_mask送任务前处理；encoder_resolution（0）接官方 TextEncodeQwenImage21.resolution，采样接该编码器的latent，避免再次独立定尺寸。
**qwen_restore_backend=swwan**：region_info接SwwanRestoreCropBoxV4，并按恢复器接口接原图/crop_box；**goohai**：region_info接原生GH_CropRestore，生成块/region_mask接对应输入。两种SEAM不可混接，不要重复逆变换。颜色还原和羽化在恢复器中配置；严格范围外保护需最终门控，抠图生成alpha不能直接当原选区处理。
查看 native_crop_width/height理解预算来源，width/height查看实际模型画布，working_resolution查看实际R。新默认用于新建节点或缺失字段；旧工作流显式保存值/已连接输入不会被强制覆盖。"""

    def _gaussian_blur_tensor(self, mask_tensor, blur_radius, processing_device):
        """PyTorch 实现的高斯模糊，支持 GPU 加速"""
        if blur_radius <= 0:
            return mask_tensor

        mask_tensor = mask_tensor.to(processing_device)
        mask_4d = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        kernel_size = int(blur_radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = blur_radius / 3.0

        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                         dtype=torch.float32, device=processing_device)
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2
        blurred = F.conv2d(mask_4d, gaussian_2d, padding=padding)
        return blurred.squeeze(0).squeeze(0)

    def _calculate_crop_box(self, mask_pil, detect, reserve_mode,
                            top_reserve, bottom_reserve, left_reserve, right_reserve,
                            top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                            reserve_max, round_to_multiple,
                            canvas_width, canvas_height, device):
        """计算 crop_box"""

        # 选择设备
        if device == "GPU":
            processing_device = model_management.get_torch_device()
        else:
            processing_device = torch.device("cpu")

        # GPU 加速的高斯模糊
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)
        blurred_tensor = self._gaussian_blur_tensor(mask_tensor, 20, processing_device)

        # 转回 PIL 进行检测（OpenCV 只能在 CPU）
        blurred_np = (blurred_tensor.cpu().numpy() * 255).astype(np.uint8)
        bluredmask = Image.fromarray(blurred_np, mode='L')

        x = 0
        y = 0
        w = 0
        h = 0

        if detect == "min_bounding_rect":
            (x, y, w, h) = min_bounding_rect(bluredmask)
        elif detect == "max_inscribed_rect":
            (x, y, w, h) = max_inscribed_rect(bluredmask)
        else:  # mask_area
            (x, y, w, h) = mask_area(mask_pil)

        # 根据 reserve_mode 计算实际的 reserve 值
        if reserve_mode == "ratio":
            # 比例模式：基于 mask 尺寸计算
            mask_width = w
            mask_height = h

            # 计算绝对像素值（左右基于宽度，上下基于高度）
            left_reserve_abs = min(mask_width * left_reserve_ratio, reserve_max)
            right_reserve_abs = min(mask_width * right_reserve_ratio, reserve_max)
            top_reserve_abs = min(mask_height * top_reserve_ratio, reserve_max)
            bottom_reserve_abs = min(mask_height * bottom_reserve_ratio, reserve_max)
        else:
            # 绝对值模式：直接使用像素值
            left_reserve_abs = left_reserve
            right_reserve_abs = right_reserve
            top_reserve_abs = top_reserve
            bottom_reserve_abs = bottom_reserve

        # 应用边距保留（使用计算后的绝对值）
        x1 = max(0, x - left_reserve_abs)
        y1 = max(0, y - top_reserve_abs)
        x2 = min(canvas_width, x + w + right_reserve_abs)
        y2 = min(canvas_height, y + h + bottom_reserve_abs)

        # 调整到指定倍数
        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            width = num_round_up_to_multiple(x2 - x1, multiple)
            height = num_round_up_to_multiple(y2 - y1, multiple)

            # 居中调整
            x1 = x1 - (width - (x2 - x1)) // 2
            y1 = y1 - (height - (y2 - y1)) // 2

            # 确保不超出边界
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            x2 = x1 + width
            y2 = y1 + height

            # 如果超出右边或底部边界，向左/上调整
            if x2 > canvas_width:
                x2 = canvas_width
                x1 = max(0, x2 - width)
            if y2 > canvas_height:
                y2 = canvas_height
                y1 = max(0, y2 - height)

        # 确保返回整数（避免 numpy.float64 导致的类型错误）
        return (int(x1), int(y1), int(x2), int(y2)), (int(x), int(y), int(w), int(h))

    def crop_by_mask_v5(self, detect, reserve_mode,
                        top_reserve, bottom_reserve, left_reserve, right_reserve,
                        top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                        reserve_max, round_to_multiple,
                        batch_mode="single_frame", device="CPU",
                        image=None, mask_image=None, crop_box=None,
                        crop_mode="bounds", mask=None, fill_mask_holes=False,
                        output_size="原像素", custom_width=1024, custom_height=1024,
                        alignment=8, edit_expansion_mode="reserve_ratio", edit_top_factor=1.2, edit_bottom_factor=1.2, edit_left_factor=1.2, edit_right_factor=1.2, qwen_resolution=1024, qwen_tolerance=1.0, qwen_restore_backend="swwan", qwen_resolution_mode="fixed", qwen_max_resolution=1024, qwen_min_resolution=128):

        if crop_mode == "qwen21_aligned":
            from .ops.qwen21_geometry import prepare_aligned
            if image is None:
                raise ValueError("Qwen aligned crop requires an image")
            if mask is None and mask_image is not None:
                mask = mask_image[:1, ..., 0]
            factors = (edit_top_factor, edit_bottom_factor, edit_left_factor, edit_right_factor) if edit_expansion_mode == "factor" else (
                1 + top_reserve_ratio, 1 + bottom_reserve_ratio, 1 + left_reserve_ratio, 1 + right_reserve_ratio)
            if fill_mask_holes:
                from .ops.qwen21_geometry import fill_holes_preserving_softness
                mask = fill_holes_preserving_softness(mask)
            prepared, region_mask, region = prepare_aligned(image, mask, qwen_resolution, qwen_tolerance, factors,
                                                              resolution_mode=qwen_resolution_mode, max_resolution=qwen_max_resolution,
                                                              min_resolution=qwen_min_resolution)
            x0,y0,x1,y1 = region["box"]
            preview = draw_rect(tensor2pil(image[:1]).convert("RGB"),x0,y0,x1-x0,y1-y0,line_color="#00FF00",line_width=2)
            width,height = region["canvas_size"]
            restore_info = region
            if qwen_restore_backend == "goohai":
                from .ops.qwen21_gh_seam import to_goohai_seam
                restore_info = to_goohai_seam(region)
            elif qwen_restore_backend != "swwan":
                raise ValueError("Unknown Qwen restore backend")
            return prepared,region_mask[...,None].repeat(1,1,1,3),list(region["box"]),pil2tensor(preview),restore_info,region_mask,width,height,0,*region["native_crop_size"],region["resolution"]

        if crop_mode == "edit_region":
            from .edit_region import EditRegionCrop

            if image is None:
                raise ValueError("Edit region requires an image")
            if mask is None and mask_image is not None:
                mask = image2mask(tensor2pil(mask_image[0:1]).convert('L'))
            if mask is None:
                raise ValueError("Edit region requires mask or mask_image")
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1:3] != image.shape[1:3]:
                raise ValueError("Edit region image and mask dimensions must match")
            if output_size == "自定义宽高" and custom_width == custom_height == 0:
                raise ValueError("At least one custom dimension must be positive")
            factors = (edit_top_factor, edit_bottom_factor, edit_left_factor, edit_right_factor) if edit_expansion_mode == "factor" else (
                1 + top_reserve_ratio, 1 + bottom_reserve_ratio, 1 + left_reserve_ratio, 1 + right_reserve_ratio)
            region, cropped, region_mask = EditRegionCrop().裁剪图像(
                image, mask, fill_mask_holes,
                *factors,
                output_size, custom_width, custom_height, alignment,
            )
            top, bottom, left, right = region["裁剪区域"]
            box = [left, top, right, bottom]
            preview = draw_rect(tensor2pil(image[0:1]).convert('RGB'),
                                left, top, right - left, bottom - top,
                                line_color="#00FF00", line_width=2)
            mask_rgb = region_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
            return cropped, mask_rgb, box, pil2tensor(preview), region, region_mask, cropped.shape[2], cropped.shape[1], 0, right-left, bottom-top, 0

        # 验证输入
        if mask_image is None and crop_box is None:
            raise ValueError("Either mask_image or crop_box must be provided")

        if image is None and mask_image is None:
            raise ValueError("At least one of image or mask_image must be provided")

        # 确定画布尺寸
        if image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
            batch_size = image.shape[0]
        elif mask_image is not None:
            canvas_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
            canvas_width, canvas_height = canvas_pil.size
            batch_size = 1
        else:
            raise ValueError("Cannot determine canvas size")

        ret_images = []
        ret_masks = []
        effective_crop_box = None
        preview_image = None

        # 处理遮罩图像（使用第一帧的 mask）
        if mask_image is not None:
            mask_pil = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('L')
            preview_image = tensor2pil(torch.unsqueeze(mask_image[0], 0)).convert('RGB')
        else:
            mask_pil = Image.new('L', (canvas_width, canvas_height), 255)
            preview_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

        # 确定使用的 crop_box（批处理模式下只计算一次）
        if crop_box is not None:
            # 优先使用输入的 crop_box
            effective_crop_box = tuple(crop_box)
            log(f"{self.NODE_NAME}: Using input crop_box: {effective_crop_box}")
        else:
            # 计算新的 crop_box（只计算一次，应用到所有帧）
            effective_crop_box, detected_rect = self._calculate_crop_box(
                mask_pil, detect, reserve_mode,
                top_reserve, bottom_reserve, left_reserve, right_reserve,
                top_reserve_ratio, bottom_reserve_ratio, left_reserve_ratio, right_reserve_ratio,
                reserve_max, round_to_multiple,
                canvas_width, canvas_height, device
            )
            log(f"{self.NODE_NAME}: Calculated crop_box: {effective_crop_box}")

            # 在预览图上绘制检测到的区域（红色）
            if detected_rect:
                x, y, w, h = detected_rect
                preview_image = draw_rect(preview_image, x, y, w, h, line_color="#FF0000",
                                          line_width=max(1, (w + h) // 100))

        # 在预览图上绘制最终裁剪框（绿色）
        preview_image = draw_rect(preview_image, effective_crop_box[0], effective_crop_box[1],
                                  effective_crop_box[2] - effective_crop_box[0],
                                  effective_crop_box[3] - effective_crop_box[1],
                                  line_color="#00FF00",
                                  line_width=max(1, (effective_crop_box[2] - effective_crop_box[0] +
                                                     effective_crop_box[3] - effective_crop_box[1]) // 200))

        # 批处理模式：应用同一个 crop_box 到所有帧
        if image is not None:
            if batch_mode == "batch_first_reuse":
                log(f"{self.NODE_NAME}: Batch mode - processing {batch_size} frames with same crop_box")

            # 裁剪 mask（首帧mask复用到所有帧）
            cropped_mask_pil = mask_pil.crop(effective_crop_box)

            for i in range(batch_size):
                img_pil = tensor2pil(torch.unsqueeze(image[i], 0)).convert('RGB')
                cropped_img = img_pil.crop(effective_crop_box)
                ret_images.append(pil2tensor(cropped_img))

                # 每帧都添加对应的 mask（复用同一个裁剪后的 mask）
                ret_masks.append(pil2tensor(cropped_mask_pil.convert('RGB')))
        else:
            # 只有 mask_image 输入的情况
            cropped_mask = mask_pil.crop(effective_crop_box)
            ret_images.append(pil2tensor(cropped_mask.convert('RGB')))
            ret_masks.append(pil2tensor(cropped_mask.convert('RGB')))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')

        return (
            torch.cat(ret_images, dim=0),
            torch.cat(ret_masks, dim=0),
            list(effective_crop_box),
            pil2tensor(preview_image),
            None,
            torch.cat(ret_masks, dim=0)[..., 0],
            ret_images[0].shape[2], ret_images[0].shape[1], 0,
            effective_crop_box[2]-effective_crop_box[0], effective_crop_box[3]-effective_crop_box[1], 0,
        )
