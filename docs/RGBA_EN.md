# RGBA processing and saving

For 1.0.0, use **Save Image (Swwan)** with RGBA Safe Pre's premultiplied RGB, alpha and has_alpha, and set `input_mode=premultiplied`. RGBA Save remains the dedicated PNG entry. RGBA Multi Save below documents the preserved Legacy interface.

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
Save Image (Swwan)   (png/webp, input_mode=premultiplied, alpha_mode=auto or keep)
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


[Project overview](../README.md)
