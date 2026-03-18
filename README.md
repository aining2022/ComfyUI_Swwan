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
| RGBA Safe Pre / Post / Save / Multi Save | 透明图安全前后处理与多格式保存 |

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
| IO Save Image Format | 轻量格式保存节点，保留质量/压缩/优化等关键参数，适合 Replicate 快速导出 |
| Patch Sage Attention KJ | 从 KJNodes 迁移的 SageAttention 模型补丁节点 |
| Get Image Size & Count | 获取图像尺寸和数量 |
| Get Latent Size & Count | 获取 Latent 尺寸和数量 |
| Preview Animation | 动画预览 |
| Fast Preview | 快速预览 |

### IO Save Image Format

这个节点从格式保存场景里裁掉了不必要的元数据和工作流导出逻辑，只保留会明显影响编码耗时和文件体积的参数：

- `quality`
- `png_compress_level`
- `optimize`
- `webp_lossless`
- `webp_method`

如果你的目标是在 Replicate 上缩短节点尾部保存耗时，优先建议：

- PNG 建议从 `png_compress_level = 1` 起步，在保存耗时和文件体积之间做平衡
- WebP 关闭 `webp_lossless`，并优先使用较低的 `webp_method`，默认值已调整为 `2`
- JPEG/TIFF 默认可保持 `optimize = true`；如果你更在意极限保存速度，也可以手动关闭

节点默认值已经偏向快速导出：

- `file_format = jpg`
- `quality = 88`
- `png_compress_level = 1`
- `optimize = true`
- `webp_method = 2`

另外，如果某些 Pillow 编码器在 `optimize = true` 时保存失败，节点会自动回退到不使用 `optimize` 再重试，避免在批处理或 Replicate 任务里直接中断。

### Patch Sage Attention KJ

这个节点已从 `ComfyUI-KJNodes` 迁移到当前仓库，用于给 `MODEL` 打上 `SageAttention` 的 attention override。

说明：

- 节点本身已内置到 `ComfyUI_Swwan`
- 运行时仍需要额外安装 `sageattention` 或 `sageattn3`
- 选择 `disabled` 时会移除当前模型上的 `optimized_attention_override`

## RGBA 节点详解

这组节点用于让带透明通道的图片安全地通过任意 RGB-only `IMAGE` 流程，并在最后按需要输出为透明 PNG / WebP 或普通 JPEG / PNG / WebP。

### 解决的问题

- PNG 透明边缘白边、黑边、发灰边
- 中间模型不支持 RGBA，只接受 RGB `IMAGE`
- 处理后 alpha 丢失
- `unpremultiply` 时因为 alpha 太小出现除 0 或爆亮
- 同一条工作流需要同时兼容透明输出和普通 RGB 输出

### 推荐工作流

保留透明输出：

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Multi Save   (png/webp, alpha_mode=auto 或 keep)
```

需要把修正后的 RGB 和 alpha 继续传给后续节点：

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Safe Post
   ↓
More Nodes
   ↓
RGBA Save / RGBA Multi Save
```

输出普通 JPEG / 无透明 PNG / 无透明 WebP：

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Multi Save   (jpeg 或 alpha_mode=flatten)
```

### 节点说明

#### RGBA Safe Pre

作用：
- 将 `Load Image` 输出的反相 `MASK` 还原成真实 alpha
- 对 RGB 执行 premultiply
- 为后面的 RGB-only 节点准备安全输入

输入：

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 输入 RGB 图像 |
| `mask` | `MASK` | `Load Image` 输出的 mask；ComfyUI 中它实际是 `1 - alpha` |

输出：

| 输出 | 类型 | 说明 |
|------|------|------|
| `image_out` | `IMAGE` | premultiplied RGB |
| `alpha` | `MASK` | 真实 alpha，范围 `[0,1]` |
| `has_alpha` | `BOOLEAN` | 当前图像是否真的带透明通道 |

说明：
- 如果输入本身没有 alpha，例如 `jpg`，这个节点会自动退化为 passthrough
- 不做 resize，不做除法，不改变尺寸

#### RGBA Safe Post

作用：
- 把 alpha resize 到处理后的图像尺寸
- 对 premultiplied RGB 做安全 `unpremultiply`
- 输出可继续传递给后续节点的修正 RGB 和 alpha

输入：

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 经过中间处理后的图像 |
| `alpha` | `MASK` | 来自 `RGBA Safe Pre` 的 alpha |
| `has_alpha` | `BOOLEAN` | 来自 `RGBA Safe Pre` 的布尔标记 |
| `epsilon` | `FLOAT` | 最小 alpha，下限保护，默认 `0.001` |

输出：

| 输出 | 类型 | 说明 |
|------|------|------|
| `image_out` | `IMAGE` | 修正后的普通 RGB |
| `alpha_out` | `MASK` | resize 后的 alpha |

什么时候需要：
- 你还要把修正后的 RGB / alpha 接给别的节点
- 你想在保存前明确看到 `Post` 处理后的结果

什么时候可以不单独接：
- 你最后直接用 `RGBA Multi Save` 输出透明 `png/webp`
- 因为 `RGBA Multi Save` 在保留透明时已经内置了同样的 `Post` 核心逻辑

#### RGBA Save

作用：
- 将 RGB 和 alpha 合并为透明 PNG
- 专用于最终透明 PNG 输出

输入：

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 要保存的 RGB 图像 |
| `alpha` | `MASK` | 要写入 PNG 的 alpha |
| `filename_prefix` | `STRING` | 输出文件名前缀 |

说明：
- 这是专用透明 PNG 输出节点
- 不支持 `jpeg` 和 `webp`
- 如果输入图像还是 premultiplied RGB，应该先经过 `RGBA Safe Post`

#### RGBA Multi Save

作用：
- 一个最终输出节点，支持 `jpeg`、`png`、`webp`
- 需要透明时内部自动执行 `RGBA Safe Post` 的核心逻辑
- 不需要透明时直接按背景色展平后保存，减少无意义的除法开销

核心输入：

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 处理后的图像 |
| `alpha` | `MASK` | 来自 `RGBA Safe Pre` 的 alpha |
| `has_alpha` | `BOOLEAN` | 来自 `RGBA Safe Pre` 的布尔标记 |
| `file_format` | `STRING` | `jpeg` / `png` / `webp` |
| `alpha_mode` | `STRING` | `auto` / `keep` / `flatten` |
| `filename_prefix` | `STRING` | 输出文件名前缀 |

可选参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `epsilon` | `FLOAT` | 仅在保留透明时使用，用于避免除 0 和爆亮 |
| `background_red` | `FLOAT` | 展平时背景色 R，默认 `1.0` |
| `background_green` | `FLOAT` | 展平时背景色 G，默认 `1.0` |
| `background_blue` | `FLOAT` | 展平时背景色 B，默认 `1.0` |
| `jpeg_quality` | `INT` | JPEG 质量，默认 `95` |
| `webp_quality` | `INT` | WebP 有损质量，默认 `90` |
| `webp_lossless` | `BOOLEAN` | 是否启用无损 WebP |
| `png_compress_level` | `INT` | PNG 压缩等级，默认 `4` |

### RGBA Multi Save 行为规则

| `file_format` | `alpha_mode` | 结果 |
|------|------|------|
| `jpeg` | 任意 | 始终展平为 RGB，JPEG 不保留透明 |
| `png` | `auto` | `has_alpha=True` 时保留透明，否则保存普通 RGB PNG |
| `png` | `keep` | 仅在 `has_alpha=True` 时保留透明 |
| `png` | `flatten` | 展平为普通 RGB PNG |
| `webp` | `auto` | `has_alpha=True` 时保留透明，否则保存普通 RGB WebP |
| `webp` | `keep` | 仅在 `has_alpha=True` 时保留透明 |
| `webp` | `flatten` | 展平为普通 RGB WebP |

### 参数选择建议

- `epsilon`
  用于 `unpremultiply` 时的下限保护，避免 `alpha=0` 或接近 0 时发生除 0 和边缘爆亮。通常保持默认 `0.001` 即可。
- `alpha_mode=auto`
  适合绝大多数工作流。`jpeg` 自动展平，`png/webp` 在有 alpha 时自动保留透明。
- `alpha_mode=keep`
  用于你明确要输出透明 `png/webp` 的情况。
- `alpha_mode=flatten`
  用于统一生成普通 RGB 文件，比如社交媒体图、缩略图、训练集 JPEG。

### 常见问答

`RGBA Multi Save` 是否包含 `RGBA Safe Post`？

是，但只在“需要保留透明”的分支里包含。

- 如果输出透明 `png/webp`，它会内部执行 alpha resize 和安全 `unpremultiply`
- 如果输出 `jpeg` 或显式选择 `flatten`，它不会执行 `Post`，而是直接按背景色合成 RGB

因此：
- 作为最终输出节点时，`RGBA Multi Save` 可以替代 `RGBA Safe Post + RGBA Save`
- 作为中间处理节点时，仍然需要单独使用 `RGBA Safe Post`

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
| RGBA Safe Pre / Post / Save / Multi Save | Safe transparent image pre/post processing and multi-format export |

## RGBA Safe Workflow

These nodes let transparent images pass safely through RGB-only `IMAGE` pipelines and then export either alpha-preserving or flattened outputs.

### Problems Solved

- white or black halos on transparent edges
- alpha loss after RGB-only processing
- divide-by-zero or blown highlights during unpremultiply
- a single workflow needing both transparent and non-transparent export formats

### Recommended Workflows

Keep transparency:

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Multi Save   (png/webp, alpha_mode=auto or keep)
```

Continue using corrected RGB plus alpha in downstream nodes:

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Safe Post
   ↓
More Nodes
   ↓
RGBA Save / RGBA Multi Save
```

Export regular RGB files:

```text
Load Image
   ↓
RGBA Safe Pre
   ↓
Any IMAGE Node
   ↓
RGBA Multi Save   (jpeg or alpha_mode=flatten)
```

### Node Details

#### RGBA Safe Pre

Purpose:
- converts ComfyUI's inverted `MASK` back to real alpha
- premultiplies RGB before RGB-only processing
- falls back to passthrough for images that do not actually contain alpha, such as JPEG inputs

Inputs:

| Parameter | Type | Description |
|------|------|------|
| `image` | `IMAGE` | input RGB image |
| `mask` | `MASK` | mask returned by `Load Image`, internally treated as `1 - alpha` |

Outputs:

| Output | Type | Description |
|------|------|------|
| `image_out` | `IMAGE` | premultiplied RGB |
| `alpha` | `MASK` | real alpha in `[0,1]` |
| `has_alpha` | `BOOLEAN` | whether the source actually contains transparency |

#### RGBA Safe Post

Purpose:
- resizes alpha to the processed image size
- safely unpremultiplies RGB
- returns corrected RGB plus alpha for downstream nodes

Inputs:

| Parameter | Type | Description |
|------|------|------|
| `image` | `IMAGE` | processed image, usually from RGB-only nodes |
| `alpha` | `MASK` | alpha returned by `RGBA Safe Pre` |
| `has_alpha` | `BOOLEAN` | boolean returned by `RGBA Safe Pre` |
| `epsilon` | `FLOAT` | lower alpha clamp used to avoid divide-by-zero and bright edges |

Outputs:

| Output | Type | Description |
|------|------|------|
| `image_out` | `IMAGE` | corrected non-premultiplied RGB |
| `alpha_out` | `MASK` | resized alpha |

Use it when:
- corrected RGB and alpha must continue into later nodes
- you want explicit control over the post step before saving

#### RGBA Save

Purpose:
- saves RGB plus alpha as transparent PNG
- dedicated final PNG node

Inputs:

| Parameter | Type | Description |
|------|------|------|
| `image` | `IMAGE` | RGB image to save |
| `alpha` | `MASK` | alpha channel to embed |
| `filename_prefix` | `STRING` | output filename prefix |

Note:
- use `RGBA Safe Post` first if your image is still premultiplied
- this node only saves PNG

#### RGBA Multi Save

Purpose:
- one final output node for `jpeg`, `png`, and `webp`
- internally performs the same safe post logic only when transparency must be preserved
- skips unnecessary unpremultiply work when flattening to RGB

Core parameters:

| Parameter | Type | Description |
|------|------|------|
| `image` | `IMAGE` | processed image |
| `alpha` | `MASK` | alpha from `RGBA Safe Pre` |
| `has_alpha` | `BOOLEAN` | boolean from `RGBA Safe Pre` |
| `file_format` | `STRING` | `jpeg`, `png`, or `webp` |
| `alpha_mode` | `STRING` | `auto`, `keep`, or `flatten` |
| `filename_prefix` | `STRING` | output filename prefix |

Optional parameters:

| Parameter | Type | Description |
|------|------|------|
| `epsilon` | `FLOAT` | used only when keeping alpha |
| `background_red` | `FLOAT` | flatten background red |
| `background_green` | `FLOAT` | flatten background green |
| `background_blue` | `FLOAT` | flatten background blue |
| `jpeg_quality` | `INT` | JPEG quality |
| `webp_quality` | `INT` | WebP lossy quality |
| `webp_lossless` | `BOOLEAN` | enable lossless WebP |
| `png_compress_level` | `INT` | PNG compression level |

### RGBA Multi Save Format Rules

| `file_format` | `alpha_mode` | Result |
|------|------|------|
| `jpeg` | any | always flattened to RGB |
| `png` | `auto` | keeps alpha only when `has_alpha=True` |
| `png` | `keep` | keeps alpha only when `has_alpha=True` |
| `png` | `flatten` | saves regular RGB PNG |
| `webp` | `auto` | keeps alpha only when `has_alpha=True` |
| `webp` | `keep` | keeps alpha only when `has_alpha=True` |
| `webp` | `flatten` | saves regular RGB WebP |

### Practical Guidance

- Keep `epsilon=0.001` unless you have a specific edge case.
- Use `alpha_mode=auto` for most workflows.
- Use `alpha_mode=keep` when you explicitly want transparent PNG/WebP output.
- Use `alpha_mode=flatten` when you want predictable RGB export for thumbnails, previews, or JPEG datasets.
- As a final output node, `RGBA Multi Save` can replace `RGBA Safe Post + RGBA Save`.
- As an intermediate node, `RGBA Safe Post` is still required because `RGBA Multi Save` does not output corrected tensors.

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
