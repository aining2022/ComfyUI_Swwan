# RGBA 处理与保存

1.0.0 推荐使用 **Save Image (Swwan)**：接 RGBA Safe Pre 的预乘 RGB、实际 alpha 和 has_alpha，设置 `input_mode=premultiplied`。RGBA Save 继续提供专用 PNG 入口。下文 RGBA Multi Save 是 Legacy 兼容接口，其参数和算法保留。

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
Save Image (Swwan)   (png/webp, input_mode=premultiplied, alpha_mode=auto 或 keep)
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


[返回项目首页](../README.md)
