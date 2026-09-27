# SPDX-License-Identifier: MIT
# Helpers adapted from ComfyUI-Apt_Preset; original notice in licenses/MIT-Apt.txt.
import torch
import numpy as np
from PIL import Image

def pil2tensor(image):  #多维度的图像也可以
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def convert_pil_image(image):
    batch_size = image.shape[0]
    converted_images = []
    for i in range(batch_size):
        single_image = image[i]  # (H, W, C)
        pil_image = tensor2pil(single_image)
        converted_image = pil2tensor(pil_image)
        converted_images.append(converted_image)
    image = torch.cat(converted_images, dim=0)
    return image
