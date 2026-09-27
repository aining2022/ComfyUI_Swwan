# 快速开始

从 README 的 21 主入口选用；低层数据合同或模型／视频补丁到 Advanced，历史工作流到 Legacy。无需安装原图片插件。

## 常用处理链

- 固定尺寸：Load Image → Resize Image (`standard`／`aspect_ratio`) → Save Image。范围限制用 Resize Range；面积限制用保留 ID 的 Resize by Megapixels。
- 多图：两个以上 IMAGE → Image Concat Multi；`strip` 保持原条带，`grid` 用 columns，`batch_grid` 将 image_1 的批次展开。inputcount 缩小遇到连接时保留端口和有效数量。
- 局部编辑：Mask Crop `edit_region` 的 IMAGE 和原生 MASK → 编辑；SEAM → Restore Crop，原图和 BOX 仍提供给原必填接口。模拟编辑即可验证位置，无需 Qwen。
- 透明图：Load Image → RGBA Safe Pre → 处理 RGB → RGBA Safe Post → RGBA Save，或直接 Save Image 显式指定 premultiplied RGB 与实际 alpha。Load Image MASK 是 `1-alpha`，不能直接当 alpha。
- 种子：Seed 的随机、递增、递减按钮设置特殊值；浏览器排队时解析，实际执行种子写入 prompt/workflow 元数据。Use last seed 恢复上次值；API 直接特殊值沿用随机回退。

## 无模型示例

`examples/cpu-image-tools.json`：纯色图 → 比例缩放 → 多输入网格 → RGB → Save Image，可直接运行。

`examples/cpu-rgba.json`：纯色 RGB 与常量 alpha → Save Image，演示直通 alpha 保存。

Qwen 的 `examples/qwen2511-remove-single-swwan.json` 只完成图片节点迁移；模型、QwenEditUtils 和推理环境仍由你提供。本轮验收未执行 Qwen。

保存相对 output_path 基于 ComfyUI output，空值直接使用 output；绝对路径按输入使用。新 saver 输出绝对路径列表。JPEG/BMP 会按背景色铺平透明度，PNG/WebP/TIFF 可保留 alpha。

BOX/SEAM 连接、批次限制及历史 ID 迁移见 [迁移说明](docs/MIGRATION.md)。
