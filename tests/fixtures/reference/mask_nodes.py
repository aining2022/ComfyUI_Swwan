# SPDX-License-Identifier: GPL-3.0-only
# Immutable upstream test reference; see THIRD_PARTY_NOTICES.md for source revisions.
class BlockifyMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "block_size": ("INT", {"default": 32, "min": 8, "max": 512, "step": 1, "tooltip": "Size of blocks in pixels (smaller = smaller blocks)"}),
                },
                "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Device to use for processing"}),
                }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = "Creates a block mask by dividing the bounding box of each mask into blocks of the specified size and filling in blocks that contain any part of the original mask."

    def process(self, masks, block_size, device="cpu"):
        processing_device = main_device if device == "gpu" else torch.device("cpu")
        
        masks = masks.to(processing_device)
        batch_size, height, width = masks.shape
        
        result_masks = torch.zeros_like(masks)
        
        for i in tqdm(range(batch_size), desc="BlockifyMask batch"):
            mask = masks[i]
            
            # Find bounding box efficiently
            mask_bool = mask > 0
            if not mask_bool.any():
                continue
                
            y_indices = torch.nonzero(mask_bool.any(dim=1), as_tuple=True)[0]
            x_indices = torch.nonzero(mask_bool.any(dim=0), as_tuple=True)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]
            
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            
            # Calculate block grid
            w_divisions = max(1, bbox_width // block_size)
            h_divisions = max(1, bbox_height // block_size)
            
            w_slice = bbox_width // w_divisions
            h_slice = bbox_height // h_divisions
            
            # Create coordinate grids only for bbox region
            y_coords = torch.arange(y_min, y_max + 1, device=processing_device).view(-1, 1)
            x_coords = torch.arange(x_min, x_max + 1, device=processing_device).view(1, -1)
            
            # Calculate block indices for bbox region
            w_block_indices = (x_coords - x_min) // w_slice
            h_block_indices = (y_coords - y_min) // h_slice
            
            # Clamp to valid range
            w_block_indices = w_block_indices.clamp(0, w_divisions - 1)
            h_block_indices = h_block_indices.clamp(0, h_divisions - 1)
            
            # Create unique block IDs by combining h and w indices
            block_ids = h_block_indices * w_divisions + w_block_indices
            
            # Get mask region within bbox
            mask_region = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Find which blocks have content using scatter_add
            max_blocks = h_divisions * w_divisions
            block_content = torch.zeros(max_blocks, device=processing_device)
            block_content.scatter_add_(0, block_ids.flatten(), mask_region.flatten())
            
            # Create result for blocks that have content
            has_content = block_content > 0
            block_mask = has_content[block_ids]
            
            # Fill the result
            result_masks[i, y_min:y_max+1, x_min:x_max+1] = block_mask.float()
        
        return (result_masks.clamp(0, 1),)
