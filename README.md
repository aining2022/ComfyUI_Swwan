# ComfyUI LayerStyle Utility Nodes

A collection of essential image processing utility nodes for ComfyUI, migrated from the popular [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) project. These nodes provide powerful image cropping, scaling, and restoration capabilities for advanced ComfyUI workflows.

## ✨ Features

- 🎯 **Smart Image Cropping**: Intelligent mask-based cropping with multiple detection modes
- 📐 **Aspect Ratio Scaling**: Flexible image scaling with aspect ratio preservation
- 🔄 **Crop Box Restoration**: Seamlessly restore cropped images back to original canvas
- 🎛️ **Fast Groups Muter**: Quick control of node groups mute/unmute state (from rgthree-comfy)
- ⚡ **Optimized Performance**: Lightweight implementation with minimal dependencies
- 🛠️ **Workflow Integration**: Designed for seamless integration in complex ComfyUI pipelines

## � Node List

### LayerUtility: CropByMask V2
Intelligently crop images based on mask regions with advanced detection algorithms.

**Features:**
- Three detection modes: `mask_area`, `min_bounding_rect`, `max_inscribed_rect`
- Customizable margin reserves (top, bottom, left, right)
- Round dimensions to multiples (8, 16, 32, 64, 128, 256, 512)
- Optional manual crop box input
- Returns cropped image, mask, crop box coordinates, and preview

**Use Cases:**
- Extract masked regions for focused processing
- Prepare images for inpainting workflows
- Optimize processing area to reduce computation

### LayerUtility: RestoreCropBox
Restore cropped images back to their original canvas position.

**Features:**
- Paste cropped images back to original coordinates
- Support for mask-based compositing
- Automatic alpha channel handling
- Batch processing support
- Mask inversion option

**Use Cases:**
- Restore processed regions to original image
- Complete crop → process → restore workflows
- Seamless image compositing

### LayerUtility: ImageScaleByAspectRatio V2
Scale images to specific aspect ratios with multiple fitting modes.

**Features:**
- Preset aspect ratios: 1:1, 3:2, 4:3, 16:9, 21:9, 3:4, 9:16, and more
- Custom aspect ratio support
- Three scaling modes: `letterbox`, `crop`, `fill`
- Scale to specific side (longest, shortest, width, height)
- Round dimensions to multiples
- SSAA (Super-Sampling Anti-Aliasing) support

**Use Cases:**
- Prepare images for specific output formats
- Maintain aspect ratios during processing
- Create consistent image dimensions for batch processing

### Fast Groups Muter (rgthree)
Quick control of workflow groups' mute/unmute state. Virtual node for UI control only.

**Features:**
- Auto-detect all groups in workflow
- Toggle mute state for all nodes in a group
- Filter groups by color or title (regex support)
- Multiple sort options (position, alphabetic, custom)
- Batch operations (mute all, enable all, toggle all)
- Quick navigation to group location

**Use Cases:**
- Quickly enable/disable entire sections of workflow
- Test different workflow branches
- Organize complex workflows with groups
- Debug by isolating specific groups

**Note:** This is a virtual node (frontend only) ported from rgthree-comfy. See `FAST_GROUPS_MUTER_README.md` for detailed usage.

### Fast Muter (rgthree)
Quick control of connected nodes' mute/unmute state. Virtual node for UI control.

**Features:**
- Auto-detect connected nodes
- Toggle mute state for each connected node
- Batch operations (mute all, enable all, toggle all)
- Toggle restrictions (default, max one, always one)

**Use Cases:**
- Control multiple nodes from a single point
- Test different processing paths
- Quickly enable/disable node chains

### Image Comparer (rgthree)
Compare two images side-by-side with interactive slider or click mode.

**Features:**
- Slide mode: hover to compare images
- Click mode: click to switch between images
- Support for image batches
- Automatic image selection from batches

**Use Cases:**
- Compare before/after processing results
- Evaluate different model outputs
- Quality control and A/B testing

### Seed (rgthree)
Enhanced seed node with special functions for randomization and control.

**Features:**
- Random seed generation
- Increment/decrement seed values
- Fixed random seed option
- Last seed tracking and reuse
- Server-side random generation fallback

**Use Cases:**
- Consistent reproducible results
- Systematic seed exploration
- Quick randomization for testing
- Seed value management

## 🚀 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "LayerStyle Utility"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
```bash
# Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/YOUR_USERNAME/ComfyUI_LayerStyle_Utility

# Install dependencies
cd ComfyUI_LayerStyle_Utility
pip install -r requirements.txt

# Restart ComfyUI
```

## 📦 Dependencies

- `torch` - PyTorch for tensor operations
- `torchvision` - Computer vision utilities
- `Pillow` - Image processing library
- `numpy` - Numerical computing
- `opencv-python` - Advanced image processing

All dependencies are automatically installed via `requirements.txt`.

## � Usage Examples

### Example 1: Crop → Process → Restore Workflow
```
[Load Image] → [CropByMask V2] → [Your Processing Node] → [RestoreCropBox] → [Save Image]
                      ↓
                  [Load Mask]
```

This workflow allows you to:
1. Crop a specific region using a mask
2. Process only the cropped area (faster, more efficient)
3. Restore the processed region back to the original image

### Example 2: Aspect Ratio Standardization
```
[Load Image] → [ImageScaleByAspectRatio V2] → [Your Model] → [Save Image]
```

Perfect for:
- Preparing images for models that require specific dimensions
- Creating consistent output sizes
- Maintaining aspect ratios during batch processing

### Example 3: Advanced Inpainting Pipeline
```
[Load Image] ──┬─→ [CropByMask V2] → [Inpainting Model] → [RestoreCropBox] ──→ [Save Image]
               │                                                    ↑
[Load Mask] ───┴────────────────────────────────────────────────────┘
```

## 🎯 Node Parameters

### CropByMask V2
- **image**: Input image tensor
- **mask**: Mask defining the crop region
- **invert_mask**: Invert the mask (default: False)
- **detect**: Detection mode (`mask_area`, `min_bounding_rect`, `max_inscribed_rect`)
- **top/bottom/left/right_reserve**: Margin pixels to add around detected region
- **round_to_multiple**: Round dimensions to specified multiple
- **crop_box** (optional): Manual crop box coordinates

### RestoreCropBox
- **background_image**: Original full-size image
- **croped_image**: Cropped image to restore
- **crop_box**: Crop box coordinates from CropByMask V2
- **croped_mask** (optional): Mask for compositing
- **invert_mask**: Invert the mask (default: False)

### ImageScaleByAspectRatio V2
- **aspect_ratio**: Target aspect ratio (original, custom, or preset)
- **proportional_width/height**: Custom aspect ratio values
- **fit**: Scaling mode (`letterbox`, `crop`, `fill`)
- **scale_to_side**: Which side to scale to (longest, shortest, width, height)
- **scale_to_length**: Target length in pixels
- **round_to_multiple**: Round dimensions to specified multiple
- **image/mask**: Input image or mask tensor

## 🛠️ Technical Details

### Detection Modes Explained

- **mask_area**: Uses the entire mask area as crop region
- **min_bounding_rect**: Finds the minimum bounding rectangle around mask
- **max_inscribed_rect**: Finds the largest rectangle that fits inside mask

### Scaling Modes Explained

- **letterbox**: Fit image within target size, add padding if needed
- **crop**: Fill target size, crop excess if needed
- **fill**: Stretch image to exactly fill target size

## 🤝 Credits

These nodes are migrated from the excellent [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) project by chflame163. We've extracted and optimized these specific utilities for users who need these functions without the full LayerStyle suite.

Original project: https://github.com/chflame163/ComfyUI_LayerStyle

## 📄 License

This project maintains the same license as the original ComfyUI_LayerStyle project.

## 🐛 Issues & Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/YOUR_USERNAME/ComfyUI_LayerStyle_Utility/issues) page
2. Create a new issue with detailed description
3. Include your ComfyUI version and error logs

## 🌟 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 Changelog

### v1.0.0 (Initial Release)
- Migrated CropByMask V2 node
- Migrated RestoreCropBox node
- Migrated ImageScaleByAspectRatio V2 node
- Created standalone utility module
- Optimized dependencies

---

**Note**: This is a focused utility package. For the complete LayerStyle suite with 100+ nodes, please visit the [original ComfyUI_LayerStyle project](https://github.com/chflame163/ComfyUI_LayerStyle).
