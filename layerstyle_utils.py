"""Utility functions extracted from ComfyUI_LayerStyle for node migration

This module contains essential image processing and conversion functions
extracted from ComfyUI_LayerStyle/py/imagefunc.py to support the migrated nodes.

Original source: https://github.com/chflame163/ComfyUI_LayerStyle
"""

import numpy as np
import torch
import cv2
import copy
from PIL import Image, ImageFilter, ImageDraw, ImageChops
from typing import Union, List
from skimage import img_as_float, img_as_ubyte


def log(message: str, message_type: str = 'info'):
    """Log messages with color coding based on message type"""
    name = 'LayerStyle'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 😺dzNodes: {name} -> {message}")


# ============================================================================
# Converter Functions
# ============================================================================

def cv22pil(cv2_img: np.ndarray) -> Image:
    """Convert OpenCV image (BGR) to PIL Image (RGB)"""
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)


def pil2cv2(pil_img: Image) -> np.array:
    """Convert PIL Image (RGB) to OpenCV image (BGR)"""
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)


def pil2tensor(image: Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert PyTorch tensor to numpy array(s)"""
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]


def tensor2pil(t_image: torch.Tensor) -> Image:
    """Convert PyTorch tensor to PIL Image"""
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def image2mask(image: Image) -> torch.Tensor:
    """Convert PIL Image to mask tensor"""
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])


def mask2image(mask: torch.Tensor) -> Image:
    """Convert mask tensor to PIL Image"""
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image


# ============================================================================
# Image Processing Functions
# ============================================================================

def gaussian_blur(image: Image, radius: int) -> Image:
    """Apply Gaussian blur to an image"""
    # image = image.convert("RGBA")
    ret_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return ret_image


def fit_resize_image(image: Image, target_width: int, target_height: int, fit: str, 
                     resize_sampler: str, background_color: str = '#000000') -> Image:
    """Resize image to target dimensions with different fit modes
    
    Args:
        image: Input PIL Image
        target_width: Target width
        target_height: Target height
        fit: Fit mode - 'letterbox', 'crop', or 'fill'
        resize_sampler: PIL resampling filter
        background_color: Background color for letterbox mode
    
    Returns:
        Resized PIL Image
    """
    image = image.convert('RGB')
    orig_width, orig_height = image.size
    if image is not None:
        if fit == 'letterbox':
            if orig_width / orig_height > target_width / target_height:  # 更宽，上下留黑
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:  # 更瘦，左右留黑
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new('RGB', size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width)//2, (target_height - fit_height)//2))
        elif fit == 'crop':
            if orig_width / orig_height > target_width / target_height:  # 更宽，裁左右
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width)//2, 0, (orig_width - fit_width)//2 + fit_width, orig_height))
            else:   # 更瘦，裁上下
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height-fit_height)//2, orig_width, (orig_height-fit_height)//2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return ret_image


def draw_rect(image: Image, x: int, y: int, width: int, height: int, 
              line_color: str, line_width: int, box_color: str = None) -> Image:
    """Draw a rectangle on an image"""
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), fill=box_color, outline=line_color, width=line_width)
    return image


# ============================================================================
# Mask Detection Functions
# ============================================================================

def mask_area(image: Image) -> tuple:
    """Detect the bounding box of non-transparent area in an image
    
    Returns:
        Tuple of (x, y, width, height)
    """
    cv2_image = pil2cv2(image.convert('RGBA'))
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    locs = np.where(thresh == 255)
    try:
        x1 = np.min(locs[1])
        x2 = np.max(locs[1])
        y1 = np.min(locs[0])
        y2 = np.max(locs[0])
    except ValueError:
        x1, y1, x2, y2 = -1, -1, 0, 0
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)


def min_bounding_rect(image: Image) -> tuple:
    """Find the minimum bounding rectangle of non-zero pixels
    
    Returns:
        Tuple of (x, y, width, height)
    """
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    x, y, width, height = 0, 0, 0, 0
    area = 0
    for contour in contours:
        _x, _y, _w, _h = cv2.boundingRect(contour)
        _area = _w * _h
        if _area > area:
            area = _area
            x, y, width, height = _x, _y, _w, _h
    return (x, y, width, height)


def max_inscribed_rect(image: Image) -> tuple:
    """Find the maximum inscribed rectangle within non-zero pixels
    
    Returns:
        Tuple of (x, y, width, height)
    """
    img = pil2cv2(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0].reshape(len(contours[0]), 2)
    rect = []
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            valid_rect = True
            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1
            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1
            if valid_rect:
                best_rect_found = True
            index_rect += 1
    # 较小的数值排前面
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)


# ============================================================================
# Color Functions
# ============================================================================

def Hex_to_RGB(inhex: str) -> tuple:
    """Convert hex color code to RGB tuple
    
    Args:
        inhex: Hex color code (e.g., '#FFFFFF' or '#FFF')
    
    Returns:
        RGB tuple (r, g, b) with values 0-255
    """
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        if len(inhex) == 4:
            inhex = "#" + "".join([char * 2 for char in inhex[1:]])
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)


# ============================================================================
# Value Functions
# ============================================================================

def is_valid_mask(tensor: torch.Tensor) -> bool:
    """Check if a mask tensor contains any non-zero values"""
    return not bool(torch.all(tensor == 0).item())


def num_round_up_to_multiple(number: int, multiple: int) -> int:
    """Round a number up to the nearest multiple

    Args:
        number: Number to round
        multiple: Multiple to round to

    Returns:
        Rounded number
    """
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        factor = (number + multiple - 1) // multiple  # 向上取整的计算方式
        return factor * multiple


# ============================================================================
# Image Blend Functions
# ============================================================================

def cv22ski(cv2_image: np.ndarray) -> np.array:
    """Convert OpenCV image to scikit-image format"""
    return img_as_float(cv2_image)


def ski2cv2(ski: np.array) -> np.ndarray:
    """Convert scikit-image to OpenCV format"""
    return img_as_ubyte(ski)


# Blend mode list
chop_mode = [
    'normal',
    'multply',
    'screen',
    'add',
    'subtract',
    'difference',
    'darker',
    'lighter',
    'color_burn',
    'color_dodge',
    'linear_burn',
    'linear_dodge',
    'overlay',
    'soft_light',
    'hard_light',
    'vivid_light',
    'pin_light',
    'linear_light',
    'hard_mix'
]


def blend_color_burn(background_image: Image, layer_image: Image) -> Image:
    """Color burn blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_2) / (img_1 + 0.001)
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))


def blend_color_dodge(background_image: Image, layer_image: Image) -> Image:
    """Color dodge blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 / (1.0 - img_1 + 0.001)
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))


def blend_linear_burn(background_image: Image, layer_image: Image) -> Image:
    """Linear burn blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2 - 1
    mask_1 = img < 0
    img = img * (1 - mask_1)
    return cv22pil(ski2cv2(img))


def blend_linear_dodge(background_image: Image, layer_image: Image) -> Image:
    """Linear dodge blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))


def blend_overlay(background_image: Image, layer_image: Image) -> Image:
    """Overlay blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))
    return cv22pil(ski2cv2(img))


def blend_soft_light(background_image: Image, layer_image: Image) -> Image:
    """Soft light blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))


def blend_hard_light(background_image: Image, layer_image: Image) -> Image:
    """Hard light blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))


def blend_vivid_light(background_image: Image, layer_image: Image) -> Image:
    """Vivid light blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 1 - (1 - img_2) / (2 * img_1 + 0.001)
    T2 = img_2 / (2 * (1 - img_1) + 0.001)
    mask_1 = T1 < 0
    mask_2 = T2 > 1
    T1 = T1 * (1 - mask_1)
    T2 = T2 * (1 - mask_2) + mask_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))


def blend_pin_light(background_image: Image, layer_image: Image) -> Image:
    """Pin light blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask_1 = img_2 < (img_1 * 2 - 1)
    mask_2 = img_2 > 2 * img_1
    T1 = 2 * img_1 - 1
    T2 = img_2
    T3 = 2 * img_1
    img = T1 * mask_1 + T2 * (1 - mask_1) * (1 - mask_2) + T3 * mask_2
    return cv22pil(ski2cv2(img))


def blend_linear_light(background_image: Image, layer_image: Image) -> Image:
    """Linear light blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 + img_1 * 2 - 1
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))


def blend_hard_mix(background_image: Image, layer_image: Image) -> Image:
    """Hard mix blend mode"""
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask = img_1 + img_2 > 1
    img = img * (1 - mask) + mask
    img = img * mask
    return cv22pil(ski2cv2(img))


def chop_image(background_image: Image, layer_image: Image, blend_mode: str, opacity: int) -> Image:
    """Apply blend mode to layer and background images

    Args:
        background_image: Background PIL Image
        layer_image: Layer PIL Image to blend
        blend_mode: Blend mode name from chop_mode list
        opacity: Opacity value (0-100)

    Returns:
        Blended PIL Image
    """
    ret_image = background_image
    if blend_mode == 'normal':
        ret_image = copy.deepcopy(layer_image)
    elif blend_mode == 'multply':
        ret_image = ImageChops.multiply(background_image, layer_image)
    elif blend_mode == 'screen':
        ret_image = ImageChops.screen(background_image, layer_image)
    elif blend_mode == 'add':
        ret_image = ImageChops.add(background_image, layer_image, 1, 0)
    elif blend_mode == 'subtract':
        ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
    elif blend_mode == 'difference':
        ret_image = ImageChops.difference(background_image, layer_image)
    elif blend_mode == 'darker':
        ret_image = ImageChops.darker(background_image, layer_image)
    elif blend_mode == 'lighter':
        ret_image = ImageChops.lighter(background_image, layer_image)
    elif blend_mode == 'color_burn':
        ret_image = blend_color_burn(background_image, layer_image)
    elif blend_mode == 'color_dodge':
        ret_image = blend_color_dodge(background_image, layer_image)
    elif blend_mode == 'linear_burn':
        ret_image = blend_linear_burn(background_image, layer_image)
    elif blend_mode == 'linear_dodge':
        ret_image = blend_linear_dodge(background_image, layer_image)
    elif blend_mode == 'overlay':
        ret_image = blend_overlay(background_image, layer_image)
    elif blend_mode == 'soft_light':
        ret_image = blend_soft_light(background_image, layer_image)
    elif blend_mode == 'hard_light':
        ret_image = blend_hard_light(background_image, layer_image)
    elif blend_mode == 'vivid_light':
        ret_image = blend_vivid_light(background_image, layer_image)
    elif blend_mode == 'pin_light':
        ret_image = blend_pin_light(background_image, layer_image)
    elif blend_mode == 'linear_light':
        ret_image = blend_linear_light(background_image, layer_image)
    elif blend_mode == 'hard_mix':
        ret_image = blend_hard_mix(background_image, layer_image)

    # Apply opacity
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)

    return ret_image
