# 快速开始指南

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI_Swwan
cd ComfyUI_Swwan
pip install -r requirements.txt
```

重启 ComfyUI 后，节点会出现在以下分类中：
- **LayerUtility**: 图片处理节点
- **rgthree**: rgthree 移植节点

## 节点快速参考

### LayerUtility 节点

#### CropByMask V2
```
[Load Image] → [Load Mask] → [CropByMask V2] → [处理节点]
                                    ↓
                              [crop_box] → [RestoreCropBox]
```

#### RestoreCropBox
```
[原始图片] ──┬─→ [RestoreCropBox] → [Save Image]
             │         ↑
[处理后图片] ─┤         │
             │         │
[crop_box] ──┴─────────┘
```

#### ImageScaleByAspectRatio V2
```
[Load Image] → [ImageScaleByAspectRatio V2] → [Save Image]
```

### rgthree 节点

#### Fast Groups Muter
```
1. 创建组: 选择节点 → 右键 → "Add Group"
2. 添加 Fast Groups Muter 节点
3. 节点会自动显示所有组的开关
```

#### Fast Muter
```
[节点A] ──┐
[节点B] ──┼─→ [Fast Muter] → [OPT_CONNECTION]
[节点C] ──┘
```

#### Image Comparer
```
[图片A] ──┐
          ├─→ [Image Comparer]
[图片B] ──┘
```

#### Seed
```
[Seed (rgthree)] → [KSampler] → [Save Image]
```

## 常见用例

### 1. 局部处理工作流
```
[Load Image] → [CropByMask V2] → [Inpaint] → [RestoreCropBox] → [Save]
     ↓              ↓                              ↑
[Load Mask] ────────┴──────────────────────────────┘
```

### 2. 多分支测试
```
创建组 "Branch A" 和 "Branch B"
↓
添加 Fast Groups Muter
↓
切换组开关测试不同分支
```

### 3. 图片对比
```
[模型A] → [Image Comparer] ← [模型B]
           ↓
      [查看对比结果]
```

### 4. 种子管理
```
[Seed] → [KSampler] → [Save]
  ↓
点击 "🎲 Randomize Each Time" 每次随机
或
点击 "🎲 New Fixed Random" 固定随机种子
```

## 快捷操作

### Fast Groups Muter
- 右键菜单 → "Mute all": 静音所有组
- 右键菜单 → "Enable all": 启用所有组
- 右键菜单 → "Toggle all": 切换所有组

### Fast Muter
- 右键菜单 → "Mute all": 静音所有连接节点
- 右键菜单 → "Enable all": 启用所有连接节点
- 右键菜单 → "Toggle all": 切换所有连接节点

### Seed
- 右键菜单 → "Randomize Each Time": 设置为随机模式
- 右键菜单 → "Use Last Queued Seed": 使用上次种子
- 右键菜单 → "Show/Hide Last Seed Value": 显示/隐藏种子值

## 属性配置

所有节点都支持右键 → "Properties" 或 "Properties Panel" 来配置高级选项。

### Fast Groups Muter 常用属性
- `matchColors`: "red,blue" - 只显示红色和蓝色的组
- `matchTitle`: "SDXL.*" - 只显示标题包含 SDXL 的组
- `sort`: "alphanumeric" - 按字母顺序排序

### Image Comparer 属性
- `comparer_mode`: "Click" - 切换到点击模式

### Seed 属性
- `randomMax`: 设置随机种子最大值
- `randomMin`: 设置随机种子最小值

## 提示和技巧

### 1. 使用组织织复杂工作流
- 将相关节点放入组中
- 使用 Fast Groups Muter 快速切换不同部分
- 用颜色区分不同功能的组

### 2. 种子探索
- 使用 Seed 节点的递增功能系统化探索
- 保存好的种子值以便后续使用
- 使用 "Last Seed" 功能快速回到之前的结果

### 3. 图片对比
- 使用 Image Comparer 对比不同参数的效果
- Slide 模式适合细节对比
- Click 模式适合整体对比

### 4. 性能优化
- 使用 CropByMask 只处理需要的区域
- 使用 Fast Muter 禁用不需要的节点
- 使用 round_to_multiple 确保尺寸符合模型要求

## 故障排除

### 节点不显示
1. 确认已重启 ComfyUI
2. 检查控制台是否有错误信息
3. 确认 `web/js` 目录存在且包含所有文件

### Fast Groups Muter 没有显示组
1. 确认已创建组（选择节点 → 右键 → "Add Group"）
2. 检查过滤条件（matchColors, matchTitle）
3. 尝试刷新节点（右键 → "Refresh"）

### Image Comparer 不显示图片
1. 确认图片已成功生成
2. 检查浏览器控制台是否有错误
3. 尝试切换 comparer_mode

### Seed 值不变
1. 检查是否设置为固定种子
2. 确认没有其他节点覆盖种子值
3. 尝试点击 "🎲 Randomize Each Time"

## 更多信息

- 详细文档: `RGTHREE_NODES_README.md`
- Fast Groups Muter 详细说明: `FAST_GROUPS_MUTER_README.md`
- 主 README: `README.md`

## 反馈和支持

如有问题或建议，请在 GitHub 上提交 Issue。
