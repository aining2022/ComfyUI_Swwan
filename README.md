# ComfyUI_Swwan

ComfyUI 自定义节点集合，收录个人常用节点，包含图像处理、Mask操作、数学运算、批处理等功能。

## 节点列表

### 图像处理 (Image)

| 节点名 | 说明 |
|--------|------|
| Image Resize KJ v2 | 多功能图像缩放，支持裁剪/填充/拉伸等模式 |
| Image Resize By Megapixels | 按目标百万像素缩放，支持宽高比控制 |
| Image Concatenate | 图像拼接（横向/纵向） |
| Image Concat From Batch | 从批次中拼接图像 |
| Image Grid Composite 2x2/3x3 | 2x2/3x3 网格合成 |
| Color Match | 颜色匹配 |
| Save Image With Alpha | 保存带透明通道的图像 |
| Cross Fade Images | 图像交叉淡入淡出 |
| Add Label | 添加文字标签 |
| Image Pad KJ | 图像填充 |
| Draw Mask On Image | 在图像上绘制 Mask |

### 图像裁剪 (Crop)

| 节点名 | 说明 |
|--------|------|
| CropByMask V2/V3 | 基于 Mask 智能裁剪 |
| RestoreCropBox | 还原裁剪区域到原图 |
| Image Crop By Mask | 按 Mask 裁剪图像 |
| Image Crop By Mask And Resize | 裁剪并缩放 |
| Image Uncrop By Mask | 还原裁剪 |

### 批处理 (Batch)

| 节点名 | 说明 |
|--------|------|
| Get Image Range From Batch | 从批次获取指定范围图像 |
| Get Images From Batch Indexed | 按索引获取图像 |
| Insert Images To Batch Indexed | 按索引插入图像 |
| Replace Images In Batch | 替换批次中的图像 |
| Shuffle Image Batch | 打乱图像顺序 |
| Reverse Image Batch | 反转图像顺序 |
| Image Batch Multi | 多图像批次合并 |
| Image List To Batch / Batch To List | 列表与批次互转 |

### 比例缩放 (Scale)

| 节点名 | 说明 |
|--------|------|
| ImageScaleByAspectRatio V2 | 按宽高比缩放 |
| Image Resize sum | 综合缩放节点 |
| Load And Resize Image | 加载并缩放图像 |

### Mask 处理

| 节点名 | 说明 |
|--------|------|
| Mask transform sum | Mask 变换 |
| NSFW Detector V2 | NSFW 内容检测 |

### 数学运算 (Math)

| 节点名 | 说明 |
|--------|------|
| Math Expression | 数学表达式计算 |
| Math Calculate | 数学计算 |
| Math Remap Data | 数值映射 |

### 开关与控制 (Switch)

| 节点名 | 说明 |
|--------|------|
| Any Switch | 任意类型切换 |
| Any Boolean Switch | 布尔切换 |

### 工具 (Utility)

| 节点名 | 说明 |
|--------|------|
| Seed | 种子节点（支持随机/递增） |
| Get Image Size & Count | 获取图像尺寸和数量 |
| Get Latent Size & Count | 获取 Latent 尺寸和数量 |
| Preview Animation | 动画预览 |
| Fast Preview | 快速预览 |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aining2022/ComfyUI_Swwan
pip install -r ComfyUI_Swwan/requirements.txt
```

## 致谢

部分节点迁移自以下开源项目：
- [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

---

# ComfyUI_Swwan (English)

Custom node collection for ComfyUI, featuring commonly used nodes for image processing, mask operations, math calculations, and batch processing.

## Node List

### Image Processing

| Node | Description |
|------|-------------|
| Image Resize KJ v2 | Multi-mode image resize with crop/pad/stretch |
| Image Resize By Megapixels | Resize by target megapixels with aspect ratio control |
| Image Concatenate | Concatenate images (horizontal/vertical) |
| Image Concat From Batch | Concatenate images from batch |
| Image Grid Composite 2x2/3x3 | 2x2/3x3 grid composition |
| Color Match | Color matching |
| Save Image With Alpha | Save image with alpha channel |
| Cross Fade Images | Image cross-fade transition |
| Add Label | Add text label |
| Image Pad KJ | Image padding |
| Draw Mask On Image | Draw mask on image |

### Image Cropping

| Node | Description |
|------|-------------|
| CropByMask V2/V3 | Smart mask-based cropping |
| RestoreCropBox | Restore cropped area to original |
| Image Crop By Mask | Crop by mask |
| Image Crop By Mask And Resize | Crop and resize |
| Image Uncrop By Mask | Restore crop |

### Batch Operations

| Node | Description |
|------|-------------|
| Get Image Range From Batch | Get image range from batch |
| Get Images From Batch Indexed | Get images by index |
| Insert Images To Batch Indexed | Insert images by index |
| Replace Images In Batch | Replace images in batch |
| Shuffle Image Batch | Shuffle image order |
| Reverse Image Batch | Reverse image order |
| Image Batch Multi | Multi-image batch merge |
| Image List To Batch / Batch To List | List-batch conversion |

### Scaling

| Node | Description |
|------|-------------|
| ImageScaleByAspectRatio V2 | Scale by aspect ratio |
| Image Resize sum | Comprehensive resize node |
| Load And Resize Image | Load and resize image |

### Mask Processing

| Node | Description |
|------|-------------|
| Mask transform sum | Mask transformation |
| NSFW Detector V2 | NSFW content detection |

### Math

| Node | Description |
|------|-------------|
| Math Expression | Math expression evaluation |
| Math Calculate | Math calculation |
| Math Remap Data | Value remapping |

### Switch & Control

| Node | Description |
|------|-------------|
| Any Switch | Any type switch |
| Any Boolean Switch | Boolean switch |

### Utility

| Node | Description |
|------|-------------|
| Seed | Seed node (random/increment) |
| Get Image Size & Count | Get image size and count |
| Get Latent Size & Count | Get latent size and count |
| Preview Animation | Animation preview |
| Fast Preview | Fast preview |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aining2022/ComfyUI_Swwan
pip install -r ComfyUI_Swwan/requirements.txt
```

## Credits

Some nodes are migrated from:
- [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
