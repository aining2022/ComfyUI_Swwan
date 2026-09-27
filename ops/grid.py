# SPDX-License-Identifier: GPL-3.0-only
"""Grid assembly and legacy exact cat layouts."""
import torch
from comfy.utils import common_upscale


def fixed_grid(images, columns):
    rows=[torch.cat(images[i:i+columns],dim=2) for i in range(0,len(images),columns)]
    return torch.cat(rows,dim=1)


def grid(images, columns, match_size=False):
    if not images:raise ValueError('Grid requires images.')
    first=images[0]; height,width=first.shape[1:3];batch=max(x.shape[0] for x in images)
    channels=max(x.shape[-1] for x in images);tiles=[]
    for image in images:
        if image.shape[1:3]!=(height,width):
            if not match_size:raise ValueError('Grid tile sizes differ; enable match_image_size.')
            image=common_upscale(image.movedim(-1,1),width,height,'lanczos','disabled').movedim(1,-1)
        image=image.to(device=first.device,dtype=first.dtype)
        if image.shape[0]<batch:image=torch.cat((image,image[-1:].repeat(batch-image.shape[0],1,1,1)))
        if image.shape[-1]<channels:image=torch.cat((image,image.new_ones((*image.shape[:-1],channels-image.shape[-1]))),dim=-1)
        tiles.append(image)
    while len(tiles)%columns:tiles.append(first.new_zeros((batch,height,width,channels)))
    return fixed_grid(tiles,columns)
