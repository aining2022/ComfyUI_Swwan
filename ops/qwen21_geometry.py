# SPDX-License-Identifier: MIT
# Size selection adapted from ComfyUI-BlueBox, copyright (c) 2026 zhoutingxi.
# Mask geometry, reversible transforms and alpha-safe restoration: Swwan, 2026.
"""One source-to-Qwen transform, shared by image, mask and inverse restoration."""
import math
from functools import lru_cache
import torch
import torch.nn.functional as F

RATIOS = (1., 4/3, 3/4, 16/9, 9/16, 3/2, 2/3)


def fixed_point(w, h, resolution):
    scale = math.sqrt(resolution ** 2 / (w * h))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


@lru_cache(maxsize=64)
def candidates(resolution, tolerance=1.):
    if resolution < 512 or resolution > 2048 or resolution % 32:
        raise ValueError('Qwen resolution must be a multiple of 32 in 512..2048')
    if not 0.2 <= tolerance <= 3.:
        raise ValueError('Qwen tolerance must be in 0.2..3')
    n = resolution // 32
    result = tuple((k*32, m*32) for k in range(1, n*3) for m in range(1, n*3)
                   if abs(k*m-n*n) <= max(1, int(n*tolerance))
                   and fixed_point(k*32, m*32, resolution) == (k*32, m*32))
    if not result:
        raise ValueError('No fixed-point canvas for this resolution')
    return result


def target_size(w, h, resolution=1024, tolerance=1.):
    ratio = w/h
    def score(wh):
        a, b = wh
        return (abs(math.log(a/b/ratio)) + .6*min(abs(math.log(a/b/r)) for r in RATIOS)
                + .4*abs(a*b-resolution**2)/resolution**2, a, b)
    return min(candidates(int(resolution), float(tolerance)), key=score)


def resize_image(image, width, height):
    if image.shape[1:3] == (height, width):
        return image
    x = image.movedim(-1, 1).float()
    if x.shape[1] == 4:
        a = x[:, 3:4]
        x = torch.cat((x[:, :3]*a, a), 1)
    x = F.interpolate(x, size=(height, width), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    if x.shape[1] == 4:
        a = x[:, 3:4]
        x = torch.cat((torch.where(a > 1e-6, x[:, :3]/a.clamp_min(1e-6), 0).clamp(0, 1), a), 1)
    return x.movedim(1, -1).to(image.dtype)


def resize_mask(mask, width, height):
    if mask.shape[-2:] == (height, width):
        return mask
    binary = bool(torch.all((mask == 0) | (mask == 1)))
    kw = {} if binary else {'align_corners': False}
    return F.interpolate(mask[:, None].float(), size=(height, width),
                         mode='nearest' if binary else 'bilinear', **kw)[:, 0].clamp(0, 1)


def normalized_mask(mask, image, allow_empty=False):
    if mask is None:
        raise ValueError('A selection mask is required')
    mask = mask.reshape(-1, *mask.shape[-2:])[:1].to(device=image.device, dtype=torch.float32).clamp(0, 1)
    if mask.shape[-2:] != image.shape[1:3]:
        raise ValueError('Image and selection mask must have the same dimensions')
    if not torch.isfinite(mask).all():
        raise ValueError('Mask contains non-finite values')
    if not allow_empty and not bool(torch.any(mask > 0)):
        raise ValueError('Mask has no selected pixels')
    return mask


def fill_holes_preserving_softness(mask):
    if mask is None:raise ValueError('A selection mask is required')
    from scipy.ndimage import binary_fill_holes
    support=mask[0].detach().cpu().numpy()>0
    holes=binary_fill_holes(support) & ~support
    return torch.maximum(mask[:1],torch.as_tensor(holes,device=mask.device,dtype=mask.dtype)[None])


def crop_area_resolution(width, height, maximum=1920):
    """Half-up rounding of native crop sqrt(area), with a legal 32-grid cap."""
    if width <= 0 or height <= 0:
        raise ValueError('Native crop dimensions must be positive')
    if maximum < 512 or maximum > 2048:
        raise ValueError('Qwen budget cap must be in 512..2048')
    legal_maximum = 32 * (int(maximum) // 32)
    rounded = 32 * math.floor(math.sqrt(width * height) / 32 + 0.5)
    return max(512, min(rounded, legal_maximum))


def prepare_aligned(image, mask, resolution=1024, tolerance=1., factors=(1.3,)*4,
                    full_image=False, resolution_mode='fixed', max_resolution=1920):
    image = image[:1]
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError('Expected one RGB or RGBA image')
    h, w = image.shape[1:3]
    if full_image:
        mask = torch.ones((1,h,w), device=image.device)
    mask = normalized_mask(mask, image)
    ys, xs = torch.where(mask[0] > 0)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    short = min(x1-x0, y1-y0)
    top, bottom, left, right = [max(0, math.ceil(short*(float(v)-1))) for v in factors]
    x0, y0, x1, y1 = max(0,x0-left), max(0,y0-top), min(w,x1+right), min(h,y1+bottom)
    native_crop_size = (x1-x0, y1-y0)
    context_box = (x0, y0, x1, y1)
    if resolution_mode == 'crop_area':
        resolution = crop_area_resolution(*native_crop_size, max_resolution)
    elif resolution_mode != 'fixed':
        raise ValueError('Unknown Qwen resolution mode')
    tw, th = target_size(*native_crop_size, resolution, tolerance)
    # Expand, never crop away a selection, to approach the chosen aspect ratio.
    vw = max(x1-x0, math.ceil((y1-y0)*tw/th))
    vh = max(y1-y0, math.ceil((x1-x0)*th/tw))
    vx0, vy0 = math.floor((x0+x1-vw)/2), math.floor((y0+y1-vh)/2)
    # Shift expansion onto available source pixels before synthesizing edge padding.
    vx0 = min(max(vx0, 0), w-vw) if vw <= w else min(0, max(w-vw,vx0))
    vy0 = min(max(vy0, 0), h-vh) if vh <= h else min(0, max(h-vh,vy0))
    vx1, vy1 = vx0+vw, vy0+vh
    bx0, by0, bx1, by1 = max(0,vx0), max(0,vy0), min(w,vx1), min(h,vy1)
    source_pad = (bx0-vx0, vx1-bx1, by0-vy0, vy1-by1)
    crop = image[:,by0:by1,bx0:bx1]
    cm = mask[:,by0:by1,bx0:bx1]
    crop = F.pad(crop.movedim(-1,1), source_pad, mode='replicate').movedim(1,-1)
    cm = F.pad(cm, source_pad, mode='constant', value=0)
    scale = min(tw/vw, th/vh)
    cw, ch = min(tw,max(1,round(vw*scale))), min(th,max(1,round(vh*scale)))
    px, py = (tw-cw)//2, (th-ch)//2
    prepared = resize_image(crop,cw,ch)
    prepared = F.pad(prepared.movedim(-1,1),(px,tw-cw-px,py,th-ch-py),mode='replicate').movedim(1,-1)
    pm = F.pad(resize_mask(cm,cw,ch),(px,tw-cw-px,py,th-ch-py))
    seam = {'qwen21_geometry_version':1, 'original_size':(w,h), 'box':(bx0,by0,bx1,by1),
            'virtual_box':(vx0,vy0,vx1,vy1), 'content_box':(px,py,px+cw,py+ch),
            'canvas_size':(tw,th), 'resolution':int(resolution), 'encoder_resolution':0,
            'native_crop_size':native_crop_size, 'context_box':context_box,
            'resolution_mode':resolution_mode, 'max_resolution':int(max_resolution),
            'source_padding':source_pad, 'original_mask':mask, 'original_image':image,
            'scale':(cw/vw,ch/vh), 'full_image':bool(full_image)}
    return prepared, pm, seam


def inverse_aligned(generated, seam, mask=False):
    tw, th = seam['canvas_size']
    actual = generated.shape[-2:] if mask else generated.shape[1:3]
    if actual != (th, tw):
        raise ValueError('Generated canvas differs from the recorded Qwen canvas')
    px,py,pr,pb = seam['content_box']
    vx0,vy0,vx1,vy1 = seam['virtual_box']
    bx0,by0,bx1,by1 = seam['box']
    if mask:
        cropped = resize_mask(generated[:,py:pb,px:pr],vx1-vx0,vy1-vy0)
        return cropped[:,by0-vy0:by1-vy0,bx0-vx0:bx1-vx0]
    cropped = resize_image(generated[:,py:pb,px:pr],vx1-vx0,vy1-vy0)
    return cropped[:,by0-vy0:by1-vy0,bx0-vx0:bx1-vx0]


def restore_aligned(background, generated, seam, cropped_mask=None, protect_mask=None,
                    feather=0, expand=0, strict=True, transparent=False):
    background = background[:1]
    w,h = seam['original_size']
    if background.shape[1:3] != (h,w):
        raise ValueError('Restore background size differs from the original')
    box = seam['box']; x0,y0,x1,y1 = box
    patch = inverse_aligned(generated, seam).to(background.device)
    if transparent:
        if patch.shape[-1] != 4:
            raise ValueError('Cutout requires native RGBA output from the Qwen VAE')
        result = patch.new_zeros((1,h,w,4))
        result[:,y0:y1,x0:x1] = patch
        return result,result[...,3]
    allowed = seam['original_mask'].to(background.device)
    if cropped_mask is None:
        m = allowed[:,y0:y1,x0:x1].clone()
    else:
        m = inverse_aligned(cropped_mask.reshape(-1, *cropped_mask.shape[-2:])[:1], seam, mask=True).to(background.device)
    if expand:
        k=2*int(expand)+1
        m=F.max_pool2d(m[:,None],k,1,k//2)[:,0]
    if feather:
        # Feather inside the selected support; never leak into untouched pixels.
        k=2*int(feather)+1
        m=F.avg_pool2d(m[:,None],k,1,k//2)[:,0]
    if strict:
        m=torch.minimum(m,allowed[:,y0:y1,x0:x1])
    if protect_mask is not None:
        p=normalized_mask(protect_mask,background,allow_empty=True)
        m=m*(1-p[:,y0:y1,x0:x1])
    result=background.clone()
    rgb=patch[...,:3]
    if background.shape[-1]==4:
        rgb=torch.cat((rgb,background[:,y0:y1,x0:x1,3:4]),-1)
    source=background[:,y0:y1,x0:x1]
    result[:,y0:y1,x0:x1]=torch.where(m[...,None]>0,source*(1-m[...,None])+rgb*m[...,None],source)
    full=allowed.new_zeros((1,h,w));full[:,y0:y1,x0:x1]=m
    return result,full
