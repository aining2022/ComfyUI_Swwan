# ComfyUI LayerStyle 实用工具节点

从流行的 [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) 项目中迁移的一组核心图像处理实用节点。这些节点为高级 ComfyUI 工作流提供强大的图像裁剪、缩放和恢复功能。

## ✨ 特性

- 🎯 **智能图像裁剪**：基于遮罩的智能裁剪，支持多种检测模式
- 📐 **宽高比缩放**：灵活的图像缩放，保持宽高比
- 🔄 **裁剪框恢复**：无缝将裁剪图像恢复到原始画布
- ⚡ **性能优化**：轻量级实现，最小化依赖
- 🛠️ **工作流集成**：专为复杂 ComfyUI 流程设计的无缝集成

## 🔧 节点列表

### LayerUtility: CropByMask V2
基于遮罩区域智能裁剪图像，支持高级检测算法。

**功能特点：**
- 三种检测模式：`mask_area`（遮罩区域）、`min_bounding_rect`（最小边界矩形）、`max_inscribed_rect`（最大内接矩形）
- 可自定义边距保留（上、下、左、右）
- 将尺寸调整为指定倍数（8、16、32、64、128、256、512）
- 可选的手动裁剪框输入
- 返回裁剪后的图像、遮罩、裁剪框坐标和预览图

**使用场景：**
- 提取遮罩区域进行集中处理
- 为修复工作流准备图像
- 优化处理区域以减少计算量

### LayerUtility: RestoreCropBox
将裁剪后的图像恢复到原始画布位置。

**功能特点：**
- 将裁剪图像粘贴回原始坐标
- 支持基于遮罩的合成
- 自动处理 alpha 通道
- 支持批量处理
- 遮罩反转选项

**使用场景：**
- 将处理后的区域恢复到原始图像
- 完成 裁剪 → 处理 → 恢复 工作流
- 无缝图像合成

### LayerUtility: ImageScaleByAspectRatio V2
将图像缩放到特定宽高比，支持多种适配模式。

**功能特点：**
- 预设宽高比：1:1、3:2、4:3、16:9、21:9、3:4、9:16 等
- 支持自定义宽高比
- 三种缩放模式：`letterbox`（信箱）、`crop`（裁剪）、`fill`（填充）
- 缩放到特定边（最长边、最短边、宽度、高度）
- 将尺寸调整为指定倍数
- SSAA（超采样抗锯齿）支持

**使用场景：**
- 为特定输出格式准备图像
- 在处理过程中保持宽高比
- 为批量处理创建一致的图像尺寸

## 🚀 安装

### 方法 1：ComfyUI Manager（推荐）
1. 打开 ComfyUI Manager
2. 搜索 "LayerStyle Utility"
3. 点击安装
4. 重启 ComfyUI

### 方法 2：手动安装
```bash
# 进入 ComfyUI 的 custom_nodes 目录
cd ComfyUI/custom_nodes

# 克隆此仓库
git clone https://github.com/YOUR_USERNAME/ComfyUI_LayerStyle_Utility

# 安装依赖
cd ComfyUI_LayerStyle_Utility
pip install -r requirements.txt

# 重启 ComfyUI
```

## 📦 依赖项

- `torch` - PyTorch 张量操作
- `torchvision` - 计算机视觉工具
- `Pillow` - 图像处理库
- `numpy` - 数值计算
- `opencv-python` - 高级图像处理

所有依赖项会通过 `requirements.txt` 自动安装。

## 📖 使用示例

### 示例 1：裁剪 → 处理 → 恢复工作流
```
[加载图像] → [CropByMask V2] → [你的处理节点] → [RestoreCropBox] → [保存图像]
                    ↓
                [加载遮罩]
```

此工作流允许你：
1. 使用遮罩裁剪特定区域
2. 仅处理裁剪区域（更快、更高效）
3. 将处理后的区域恢复到原始图像

### 示例 2：宽高比标准化
```
[加载图像] → [ImageScaleByAspectRatio V2] → [你的模型] → [保存图像]
```

适用于：
- 为需要特定尺寸的模型准备图像
- 创建一致的输出尺寸
- 在批量处理期间保持宽高比

### 示例 3：高级修复流程
```
[加载图像] ──┬─→ [CropByMask V2] → [修复模型] → [RestoreCropBox] ──→ [保存图像]
             │                                              ↑
[加载遮罩] ───┴──────────────────────────────────────────────┘
```

## 🎯 节点参数

### CropByMask V2
- **image**：输入图像张量
- **mask**：定义裁剪区域的遮罩
- **invert_mask**：反转遮罩（默认：False）
- **detect**：检测模式（`mask_area`、`min_bounding_rect`、`max_inscribed_rect`）
- **top/bottom/left/right_reserve**：在检测区域周围添加的边距像素
- **round_to_multiple**：将尺寸调整为指定倍数
- **crop_box**（可选）：手动裁剪框坐标

### RestoreCropBox
- **background_image**：原始全尺寸图像
- **croped_image**：要恢复的裁剪图像
- **crop_box**：来自 CropByMask V2 的裁剪框坐标
- **croped_mask**（可选）：用于合成的遮罩
- **invert_mask**：反转遮罩（默认：False）

### ImageScaleByAspectRatio V2
- **aspect_ratio**：目标宽高比（original、custom 或预设）
- **proportional_width/height**：自定义宽高比值
- **fit**：缩放模式（`letterbox`、`crop`、`fill`）
- **scale_to_side**：缩放到哪一边（longest、shortest、width、height）
- **scale_to_length**：目标长度（像素）
- **round_to_multiple**：将尺寸调整为指定倍数
- **image/mask**：输入图像或遮罩张量

## 🛠️ 技术细节

### 检测模式说明

- **mask_area**：使用整个遮罩区域作为裁剪区域
- **min_bounding_rect**：找到遮罩周围的最小边界矩形
- **max_inscribed_rect**：找到适合遮罩内部的最大矩形

### 缩放模式说明

- **letterbox**：将图像适配到目标尺寸内，必要时添加填充
- **crop**：填充目标尺寸，必要时裁剪多余部分
- **fill**：拉伸图像以完全填充目标尺寸

## 🤝 致谢

这些节点从优秀的 [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) 项目（作者：chflame163）迁移而来。我们提取并优化了这些特定的实用工具，供需要这些功能但不需要完整 LayerStyle 套件的用户使用。

原始项目：https://github.com/chflame163/ComfyUI_LayerStyle

## 📄 许可证

本项目保持与原始 ComfyUI_LayerStyle 项目相同的许可证。

## 🐛 问题与支持

如果遇到任何问题或有疑问：
1. 查看 [Issues](https://github.com/YOUR_USERNAME/ComfyUI_LayerStyle_Utility/issues) 页面
2. 创建新问题并提供详细描述
3. 包含你的 ComfyUI 版本和错误日志

## 🌟 贡献

欢迎贡献！请随时：
- 报告错误
- 建议新功能
- 提交拉取请求
- 改进文档

## 📝 更新日志

### v1.0.0（初始版本）
- 迁移 CropByMask V2 节点
- 迁移 RestoreCropBox 节点
- 迁移 ImageScaleByAspectRatio V2 节点
- 创建独立实用工具模块
- 优化依赖项

---

**注意**：这是一个专注的实用工具包。如需完整的 LayerStyle 套件（100+ 节点），请访问[原始 ComfyUI_LayerStyle 项目](https://github.com/chflame163/ComfyUI_LayerStyle)。
