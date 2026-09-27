# SPDX-License-Identifier: GPL-3.0-only
from .image_common import F, math, torch

def crossfade(images_1, images_2, alpha):
    crossfade = (1 - alpha) * images_1 + alpha * images_2
    return crossfade

def ease_in(t):
    return t * t

def ease_out(t):
    return 1 - (1 - t) * (1 - t)

def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t

def bounce(t):
    if t < 0.5:
        return ease_out(t * 2) * 0.5
    else:
        return ease_in((t - 0.5) * 2) * 0.5 + 0.5

def elastic(t):
    return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))

def glitchy(t):
    return t + 0.1 * math.sin(40 * t)

def exponential_ease_out(t):
    return 1 - (1 - t) ** 4

easing_functions = {
    "linear": lambda t: t,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
    "elastic": elastic,
    "glitchy": glitchy,
    "exponential_ease_out": exponential_ease_out,
}

def transition_images(images_1, images_2, alpha, transition_type, blur_radius, reverse):
    width = images_1.shape[1]
    height = images_1.shape[0]

    mask = torch.zeros_like(images_1, device=images_1.device)

    alpha = alpha.item()
    if reverse:
        alpha = 1 - alpha

    #transitions from matteo's essential nodes
    if "horizontal slide" in transition_type:
        pos = round(width * alpha)
        mask[:, :pos, :] = 1.0
    elif "vertical slide" in transition_type:
        pos = round(height * alpha)
        mask[:pos, :, :] = 1.0
    elif "box" in transition_type:
        box_w = round(width * alpha)
        box_h = round(height * alpha)
        x1 = (width - box_w) // 2
        y1 = (height - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        mask[y1:y2, x1:x2, :] = 1.0
    elif "circle" in transition_type:
        radius = math.ceil(math.sqrt(pow(width, 2) + pow(height, 2)) * alpha / 2)
        c_x = width // 2
        c_y = height // 2
        x = torch.arange(0, width, dtype=torch.float32, device="cpu")
        y = torch.arange(0, height, dtype=torch.float32, device="cpu")
        y, x = torch.meshgrid((y, x), indexing="ij")
        circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
        mask[circle] = 1.0
    elif "horizontal door" in transition_type:
        bar = math.ceil(height * alpha / 2)
        if bar > 0:
            mask[:bar, :, :] = 1.0
            mask[-bar:,:, :] = 1.0
    elif "vertical door" in transition_type:
        bar = math.ceil(width * alpha / 2)
        if bar > 0:
            mask[:, :bar,:] = 1.0
            mask[:, -bar:,:] = 1.0
    elif "fade" in transition_type:
        mask[:, :, :] = alpha

    mask = gaussian_blur(mask, blur_radius)

    return images_1 * (1 - mask) + images_2 * mask

def gaussian_blur(mask, blur_radius):
    if blur_radius > 0:
        kernel_size = int(blur_radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        sigma = blur_radius / 3
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        x = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(mask.device)
        kernel2d = kernel2d.expand(mask.shape[2], 1, kernel2d.shape[0], kernel2d.shape[1])
        mask = mask.permute(2, 0, 1).unsqueeze(0)  # Change to [C, H, W] and add batch dimension
        mask = F.conv2d(mask, kernel2d, padding=kernel_size // 2, groups=mask.shape[1])
        mask = mask.squeeze(0).permute(1, 2, 0)  # Change back to [H, W, C]
    return mask

easing_functions = {
    "linear": lambda t: t,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": bounce,
    "elastic": elastic,
    "glitchy": glitchy,
    "exponential_ease_out": exponential_ease_out,
}
