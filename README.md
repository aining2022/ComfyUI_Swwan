# ComfyUI_Swwan

ComfyUI 的图片、遮罩与工作流工具。通过推荐入口和模式切换减少重复选择，同时保留历史节点的兼容接口。

当前有 **21 个推荐入口**、77 个任务专用工具、18 个 Legacy 兼容节点和 2 个实验功能，共 118 个节点。新工作流优先从推荐入口选择；已有工作流按迁移说明保留行为。

**使用者从本页开始。AI agent 请先读 [AGENTS.md](AGENTS.md)。**

## 安装与更新

需要 **Python ≥3.11** 和可正常运行的 ComfyUI。以下 `python` 必须是启动 ComfyUI 使用的解释器；已有 torch/CUDA 环境继续使用。

```sh
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/aining2022/ComfyUI_Swwan.git
cd ComfyUI_Swwan
python -m pip install -r requirements.txt
```

Windows portable 版可在其根目录安装依赖：

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_Swwan\requirements.txt
```

安装后重启 ComfyUI，刷新浏览器，在节点搜索中输入 `Swwan`。本仓对应的图片处理能力无需安装 KJNodes、LayerStyle、rgthree、孤海等原插件。

| 使用范围 | 依赖 |
| --- | --- |
| 基础注册 | ComfyUI 环境已有的 torch、numpy、Pillow |
| 常用图片／遮罩算法 | `requirements.txt` 安装 OpenCV、SciPy、scikit-image；torchvision 使用与现有 torch 匹配的 ComfyUI 环境版本 |
| Color Match 的 `color_matcher` 模式 | 另执行 `python -m pip install color-matcher` |
| Color Match 的 `mean_std` 等模式 | 使用 ComfyUI 的 kornia；缺失时在同一环境补装 |
| RTX VSR／SageAttention／设备功能 | 按 [设备与模型功能](docs/ADVANCED.md) 配置对应硬件和库 |

可选依赖在使用对应功能时加载。项目不自动安装 CUDA wheels，也不自动下载模型。

更新时在仓库目录执行，随后重启 ComfyUI 并刷新浏览器：

```sh
git pull --ff-only
python -m pip install -r requirements.txt
```

如果存在本地修改或分支分歧导致更新失败，先保存并处理自己的改动。

## 第一个工作流

先导入 [无模型图片示例](examples/cpu-image-tools.json)，执行“纯色图 → 比例缩放 → 网格拼接 → RGB → 保存”。它只需要本仓与 ComfyUI 核心节点，可用于检查安装。

[透明图示例](examples/cpu-rgba.json) 演示 RGB 配合实际 alpha 保存。更多连接方式见 [快速开始](QUICK_START.md)。

## 按任务选节点

以下是 21 个推荐入口。表中的名称对应节点搜索显示名，均带 `(Swwan)`；保存的节点 ID、完整端口和旧版替代关系见 [节点目录](docs/NODE_CATALOG.md)。

| 要做什么 | 推荐节点 | 怎么选择 |
| --- | --- | --- |
| 按遮罩裁剪，再放回原图 | Mask Crop、Restore Crop | 普通边界裁剪用 `bounds`；局部编辑用 `edit_region`，配套连接 SEAM 还原 |
| 指定宽高或图片比例 | Resize Image | `standard` 用常规尺寸规则，`aspect_ratio` 用比例预设；迁移的孤海／Essentials 算法分别用 `edit_size`／`essentials` |
| 控制总像素量 | Image Resize By Megapixels | 按百万像素约束图片面积 |
| 限制最小／最大尺寸 | Image Resize Range | 按尺寸上下限缩放，也支持只输入遮罩 |
| 图层混合或按遮罩合成 | Image Blend | 默认模式用于图层混合；`mask_composite` 提供尺寸匹配、遮罩扩缩和模糊 |
| 多图拼接或批次网格 | Image Concat Multi | `strip` 为条带，`grid` 为多个输入排网格，`batch_grid` 为首输入的批次排网格 |
| 生成纯色图、转换颜色参数或转 RGB | ColorImage、Color Converter、Images to RGB | 分别输出图片、颜色格式和 RGB 图片；Color Converter 可输出 COLORCODE |
| 将遮罩按块归整 | Blockify Mask | 独立的遮罩块化任务 |
| 转换或截取图片批次 | Image List to Image Batch、Image Batch to Image List、Get Image Range From Batch | list 可包含不同尺寸；tensor batch 中的图片必须同尺寸 |
| 处理和保存透明图片 | RGBA Safe Pre、RGBA Safe Post、RGBA Save | Pre/Post 配套处理实际 alpha，Save 是专用 PNG 保存入口 |
| 保存普通或透明图片 | Save Image | 支持多格式、路径、元数据、字幕和工作流 JSON |
| 数学计算、条件切换、种子控制 | Math Expression、Any Boolean Switch、Seed | 分别用于安全表达式、选择输入、随机／递增／递减及上次种子 |

需要更细的操作时到 `Swwan/Advanced/<任务>`。Color Match、Color Shift Fix、BBOX、IMAGE_BOUNDS、视频和模型补丁各有独立用途。`Swwan/Legacy` 用于历史兼容，`Swwan/Experimental` 包含设备实验功能。

扩图使用 `Image Pad For Outpaint Masked (Swwan)`：`padding_mode=directional` 支持 `padding_unit` 在像素／百分比之间切换，左右按原图宽度、上下按原图高度计算；`alignment` 控制尺寸对齐，`feathering` 为高斯羽化。该模式处理单张 RGB 图；默认 `legacy` 保留原像素扩边与批次行为。四侧均为 0 且启用对齐时，按原扩图算法向下裁齐。

常用遮罩专用工具如下，它们不运行检测或分割模型：

| 工具 | 用途 |
| --- | --- |
| Mask Process | 清理、填孔、二值化、扩缩和模糊；不同模式保留各自算法 |
| Mask Combine | 遮罩运算、边界取景与对齐 |
| Mask Analyze | 区域尺寸、位置和面积检测 |
| Mask Segments | MASK 与 SEGS 转换，以及区域排序／筛选 |
| Image Matte | 用已有遮罩抠图，输出透明 RGBA、填色 RGB 和 MASK |

## 连接与保存时的约定

- **局部编辑**：Mask Crop 的 `edit_region` 处理首张图片。输出的 IMAGE 和末尾原生 MASK 接处理链，SEAM 接 Restore Crop；Restore 的原图和 BOX 仍需连接。普通 `bounds` 模式保留既有批次规则。
- **遮罩与透明度**：ComfyUI Load Image 的 MASK 是 `1 - alpha`；RGBA Safe Pre 输出的是实际 alpha。保存时按输入选择 straight 或 premultiplied RGB，详见 [RGBA 说明](docs/RGBA.md)。
- **裁剪信息**：BOX、BBOX、IMAGE_BOUNDS、SEAM、STITCH3 各有不同协议，不能直接互换。
- **保存路径**：Save Image 的相对路径基于 ComfyUI output，绝对路径按输入使用。每个目录独立编号，不覆盖已有图片或同名字幕／JSON；输出绝对路径列表，output 内文件提供 UI 预览。
- **格式**：PNG／WebP／TIFF 可保留 alpha；JPEG／BMP 按背景色铺平透明度。RGBA Save 保留专用 PNG 入口；Legacy 保存节点继续遵循各自的历史接口。

## 示例与迁移

| 示例 | 用途与运行条件 |
| --- | --- |
| [CPU 图片工具](examples/cpu-image-tools.json) | 无模型，可直接检查缩放、拼接、RGB 和保存 |
| [CPU 透明图](examples/cpu-rgba.json) | 无模型，演示实际 alpha 保存 |
| [Qwen2511 移除](examples/qwen2511-remove-single-swwan.json) | 图片处理节点已迁移；仍需对应模型、QwenEditUtils 和推理环境 |
| [Qwen 换脸／换头](examples/qwen-face-head-swwan.json) | 43 个纯图片／遮罩实例已迁移；RMBG、人体分割、SAM3、DW 姿态、人脸对齐、循环控制、LG 颜色节点及模型依赖保留，见 [对照说明](docs/QWEN_FACE_HEAD_INTEGRATION.md) |

1.0.0 为重名节点分配了独立 ID，不注册原插件冲突别名。**显示名相似不代表旧节点 ID 可直接使用。** 通用迁移工具支持 UI 工作流和 API prompt，默认另存并保留原文件：

```sh
python scripts/migrate_workflow.py old.json --dry-run
python scripts/migrate_workflow.py old.json --swwan-node-id 12 --swwan-node-id 35 --comfyui-root /path/to/ComfyUI
```

`12`、`35` 是示意编号，请替换为属于 Swwan 的节点。来源不明的重名节点需要显式选择，只有确认整个文件属于本仓历史工作流时才使用 `--assume-swwan`。Qwen 专用迁移命令及数据合同见 [迁移说明](docs/MIGRATION.md) 和 [换脸／换头说明](docs/QWEN_FACE_HEAD_INTEGRATION.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 搜索不到 `(Swwan)` 节点 | 确认目录在 `ComfyUI/custom_nodes/ComfyUI_Swwan`，检查启动日志及 Python 环境，重启服务 |
| 更新后控件仍旧或显示异常 | 刷新浏览器；必要时强制刷新。检查网络中的 `/extensions/ComfyUI_Swwan/` 脚本是否加载成功 |
| 旧工作流出现缺失节点 | 先核对旧 ID 与来源，运行迁移 dry-run；模型或控制类第三方节点仍需原插件 |
| 节点执行提示依赖缺失 | 在启动 ComfyUI 的同一 Python 环境安装错误提示中的库；Color Match 的旧模式另需 color-matcher |
| 遮罩或透明度反了 | 检查输入是编辑 MASK、Load Image 的 `1-alpha`，还是实际 alpha，再选择反转／保存参数 |
| list 转 batch 失败 | 空 list 不可转换；尺寸不同的图片先统一尺寸，或继续使用 list |
| 局部编辑批次只处理首张 | `edit_region` 的约定是首张处理；逐张编辑需在工作流中逐张调用 |

## 验证、贡献与许可

本项目提供 CPU 图片／遮罩参考对照、历史接口回归、实际保存文件和前端检查。**CPU 通过不代表 CUDA、RTX、检测模型或 Qwen 生成效果通过。** 已验证环境和剩余项见 [验收报告](docs/PROJECT_REVIEW.md)、[安装检查](INSTALLATION_CHECKLIST.md) 与 [硬件验收](docs/HARDWARE_VALIDATION.md)。

开发流程和命令见 [AI agent 入口](AGENTS.md)、[贡献指南](CONTRIBUTING.md) 和 [节点开发 skill](skills/comfyui-swwan-node-development/SKILL.md)。变更记录见 [CHANGELOG](CHANGELOG.md)。

整包按 **GPL-3.0** 分发；各来源文件保留原许可、版权和修改说明。详见 [LICENSE](LICENSE)、[来源与资产声明](THIRD_PARTY_NOTICES.md)。
