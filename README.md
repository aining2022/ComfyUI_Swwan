# ComfyUI_Swwan

独立的 ComfyUI 图片处理与工作流工具。1.0.0 保留全部历史能力，用 **21 个推荐主入口**、任务专用工具和 Legacy 兼容层降低选择成本。当前注册 **113 个节点**：21 主入口、72 专用工具、18 Legacy、2 Experimental。

## 从任务选择节点

| 推荐入口 | 节点 ID | 菜单 |
| --- | --- | --- |
| Mask Crop (Swwan) | `SwwanCropByMaskV5` | Swwan/Image |
| Restore Crop (Swwan) | `SwwanRestoreCropBoxV4` | Swwan/Image |
| ColorImage (Swwan) | `LayerUtility: ColorImage (Swwan)` | Swwan/Image |
| Image Blend (Swwan) | `ImageBlendSwwan` | Swwan/Image |
| Seed (Swwan) | `SwwanSeed` | Swwan/Utils |
| Image List to Image Batch (Swwan) | `SwwanImageListToImageBatch` | Swwan/Batch |
| Image Batch to Image List (Swwan) | `SwwanImageBatchToImageList` | Swwan/Batch |
| Get Image Range From Batch (Swwan) | `SwwanGetImageRangeFromBatch` | Swwan/Batch |
| Image Concat Multi (Swwan) | `SwwanImageConcatMulti` | Swwan/Image |
| Resize Image (Swwan) | `ImageResizeKJv2Alternative` | Swwan/Image |
| Image Resize By Megapixels (Swwan) | `ImageResizeByMegapixels` | Swwan/Image |
| Any Boolean Switch (Swwan) | `AnyBooleanSwitch (Swwan)` | Swwan/Utils |
| Math Expression (Swwan) | `MathExpression_UTK` | Swwan/Utils |
| RGBA Safe Pre (Swwan) | `RGBA_Safe_Pre` | Swwan/RGBA |
| RGBA Safe Post (Swwan) | `RGBA_Safe_Post` | Swwan/RGBA |
| RGBA Save (Swwan) | `RGBA_Save` | Swwan/RGBA |
| Image Resize Range (Swwan) | `SwwanImageResizeRange` | Swwan/Image |
| Blockify Mask (Swwan) | `SwwanBlockifyMask` | Swwan/Mask |
| Images to RGB (Swwan) | `SwwanImagesToRGB` | Swwan/Image |
| Color Converter (Swwan) | `SwwanColorConverter` | Swwan/Image |
| Save Image (Swwan) | `SwwanSaveImage` | Swwan/IO |

三类缩放分别解决固定尺寸／比例、百万像素、尺寸上下限。Image Blend 的图层混合与遮罩合成通过模式切换。Concat 的 `strip` 保留原条带算法，`grid` 为多输入网格，`batch_grid` 将首输入中的批次铺成网格。Color Shift Fix、BBOX、IMAGE_BOUNDS、视频与模型补丁等按任务放在 `Swwan/Advanced`，用途不同的能力继续独立。

[全部节点与替代关系](docs/NODE_CATALOG.md) 由唯一清单 [node_manifest.json](node_manifest.json) 和实际接口生成。[整改验收报告](docs/PROJECT_REVIEW.md) 记录完成证据与限制。

## 安装

在 ComfyUI 的 Python 环境中安装本仓 `requirements.txt`，把仓库放进 `ComfyUI/custom_nodes`，重启服务并刷新浏览器。基础注册只需 ComfyUI 已有的 torch、numpy、Pillow；图像算法依赖按 `pyproject.toml` 的 vision/color 分层。Color Match 执行时另需 `color-matcher`。GPU 专用库按硬件安装，本项目不自动安装 CUDA wheels。

```sh
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/aining2022/ComfyUI_Swwan.git
cd ComfyUI_Swwan
python -m pip install -r requirements.txt
```

无需安装 KJNodes、LayerStyle、rgthree 或孤海插件即可注册和使用本仓对应能力。前端仅加载本仓 5 个脚本。

## 历史工作流迁移

1.0.0 为 56 个 KJ 重名 ID、4 个 LayerStyle 重名 ID 和种子分配独立 ID；其余既有 ID 保留，包含 `ImageResizeByMegapixels`。不注册冲突别名。**显示名称变化不等于 ID 兼容**，旧重名节点需要迁移。

通用工具支持 UI workflow 和 API prompt，默认另存。来源不明的同名节点必须指定属于 Swwan 的节点编号，未选择的第三方节点保持原值：

```sh
python scripts/migrate_workflow.py old.json --dry-run
python scripts/migrate_workflow.py old.json --swwan-node-id 12 --swwan-node-id 35 --comfyui-root /path/to/ComfyUI
```

只有已确认整个文件来自本仓时使用 `--assume-swwan`。工具拒绝覆盖原文件，已存在的目标文件也不会覆盖。迁移来源和接口详见 [迁移说明](docs/MIGRATION.md)。

## 图片、遮罩与保存

Mask Crop 的 `bounds` 保留原行为；`edit_region` 处理首帧，提供填孔、最短边比例扩展、自定义输出与越界填充。原四个输出保留，末尾为 `SEAM`、原生 `MASK`；Restore Crop 有接缝时执行编辑还原，否则执行原 BOX 分支。

`BOX`、`BBOX`、`IMAGE_BOUNDS`、`SEAM`、`STITCH3` 不能直接互换。ComfyUI Load Image 的 MASK 表示 `1-alpha`，RGBA Safe Pre 输出实际 alpha；保存时显式区分 straight 和 premultiplied RGB。

Save Image 统一 PNG/JPEG/WebP/TIFF/BMP、alpha、元数据、字幕与 workflow JSON。相对路径基于 ComfyUI output，绝对路径按输入使用；返回绝对路径列表，output 内文件提供 UI 预览。每个目录独立编号，不覆盖既有图片或同名 sidecar。RGBA Save 保留专用 PNG 入口；旧保存节点保留历史路径和命名合同。

- [快速开始与无模型示例](QUICK_START.md)
- [RGBA 约定](docs/RGBA.md) / [English](docs/RGBA_EN.md)
- [设备功能](docs/ADVANCED.md) / [硬件验收](docs/HARDWARE_VALIDATION.md)

Qwen2511 的 [迁移工作流](examples/qwen2511-remove-single-swwan.json) 保留核心节点、QwenEditUtils、模型、提示词和采样配置。执行该工作流仍需对应模型和 QwenEditUtils。

## 维护与验证

[贡献指南](CONTRIBUTING.md)、[变更说明](CHANGELOG.md)、[安装验收清单](INSTALLATION_CHECKLIST.md)。仓库维护的 [节点开发 skill](skills/comfyui-swwan-node-development/SKILL.md) 查询当前机器可读目录，判断复用、扩展、优化或新增，不复制节点名单。

```sh
python scripts/export_node_catalog.py --comfyui-root /path/to/ComfyUI
python tests/test_workflow_image_nodes.py --comfyui-root /path/to/ComfyUI
python tests/test_project_contracts.py --comfyui-root /path/to/ComfyUI
python tests/test_optional_imports.py --comfyui-root /path/to/ComfyUI
node tests/test_workflow_image_modes.mjs
node tests/test_project_frontend.mjs
python tests/test_skill_candidates.py
```

CPU CI 固定 ComfyUI `4e024cb1e8423be735b919c585669dc189735c7b`，参考实现随仓库提供。本机验证范围与 GPU 待验项分开记录，不把 CPU 通过作为 CUDA 或生成效果通过。

整包 **GPL-3.0** 分发；MIT 等来源文件保留原许可及版权。[完整来源及资产声明](THIRD_PARTY_NOTICES.md)。
