# SPDX-License-Identifier: GPL-3.0-only
# Pure SEGS processing derived from ltdrdata/ComfyUI-Impact-Pack; no detector/model imports.
import logging
from collections import namedtuple
import numpy as np
import torch

SEG = namedtuple("SEG", ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"], defaults=[None])

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]

def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None, is_contour=True):
    from .dependencies import lazy_module
    cv2 = lazy_module("cv2", "vision")
    drop_size = max(drop_size, 1)
    if mask is None:
        logging.info("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    if isinstance(mask, np.ndarray):
        pass  # `mask` is already a NumPy array
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            logging.info("[mask_to_segs] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)

    if mask is None:
        logging.info("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    for i in range(mask.shape[0]):
        mask_i = mask[i]

        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (
                    np.min(indices[1]),
                    np.min(indices[0]),
                    np.max(indices[1]),
                    np.max(indices[0]),
                )
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor
                )
                x1, y1, x2, y2 = crop_region

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]

                    if bbox_fill:
                        bx1, by1, bx2, by2 = bbox
                        cropped_mask = cropped_mask.copy()
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        item = SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None)
                        result.append(item)

        else:
            mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
            contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                hierarchy = ctree[0][j]
                if hierarchy[3] != -1:
                    continue

                separated_mask = np.zeros_like(mask_i_uint8)
                cv2.drawContours(separated_mask, [contour], 0, 255, -1)
                separated_mask = np.array(separated_mask / 255.0).astype(np.float32)

                x, y, w, h = cv2.boundingRect(contour)
                bbox = x, y, x + w, y + h
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor, crop_min_size
                )

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if w > drop_size and h > drop_size:
                    if is_contour:
                        mask_src = separated_mask
                    else:
                        mask_src = mask_i * separated_mask

                    cropped_mask = np.array(
                        mask_src[
                            crop_region[1]: crop_region[3],
                            crop_region[0]: crop_region[2],
                        ]
                    )

                    if bbox_fill:
                        cx1, cy1, _, _ = crop_region
                        bx1 = x - cx1
                        bx2 = x+w - cx1
                        by1 = y - cy1
                        by2 = y+h - cy1
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        cropped_mask = torch.clip(torch.from_numpy(cropped_mask), 0, 1.0)
                        item = SEG(None, cropped_mask.numpy(), 1.0, crop_region, bbox, label, None)
                        result.append(item)

    if not result:
        logging.info("[mask_to_segs] Empty mask.")

    logging.info(f"# of Detected SEGS: {len(result)}")
    # for r in result:
    #     print(f"\tbbox={r.bbox}, crop={r.crop_region}, label={r.label}")

    # shape: (b,h,w) -> (h,w)
    return (mask.shape[1], mask.shape[2]), result

def segs_to_combined_mask(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        crop_region = seg.crop_region
        mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

    return torch.from_numpy(mask.astype(np.float32) / 255.0)

def segs_to_masklist(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]

    masks = []
    for seg in segs[1]:
        if isinstance(seg.cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(seg.cropped_mask)
        else:
            cropped_mask = seg.cropped_mask

        if cropped_mask.ndim == 2:
            cropped_mask = cropped_mask.unsqueeze(0)

        n = len(cropped_mask)

        mask = torch.zeros((n, h, w), dtype=torch.uint8)
        crop_region = seg.crop_region
        mask[:, crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).to(torch.uint8)
        mask = (mask / 255.0).to(torch.float32)

        for x in mask:
            masks.append(x)

    if len(masks) == 0:
        empty_mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
        masks = [empty_mask]

    return masks
