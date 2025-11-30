# Fast Groups Muter 节点移植说明

## 概述
Fast Groups Muter 是从 rgthree-comfy 移植的节点，用于快速控制 ComfyUI 工作流中组（Group）内所有节点的静音/激活状态。

## 文件结构

```
ComfyUI_Swwan/
├── fast_groups_muter.py          # Python 后端（虚拟节点，仅注册）
├── web/js/
│   ├── fast_groups_muter_entry.js  # 简化版入口文件（推荐使用）
│   ├── fast_groups_muter.js        # 完整版（需要更多依赖）
│   ├── utils.js                    # 工具函数
│   ├── utils_canvas.js             # Canvas 绘制工具
│   ├── utils_widgets.js            # Widget 工具
│   ├── base_node.js                # 基础节点类
│   ├── constants.js                # 常量定义
│   ├── rgthree.js                  # rgthree 核心
│   ├── common/
│   │   ├── dialog.js               # 对话框组件
│   │   └── shared_utils.js         # 共享工具
│   └── services/
│       ├── fast_groups_service.js  # 组管理服务
│       └── key_events_services.js  # 键盘事件服务
└── __init__.py                     # 节点注册
```

## 使用方法

### 1. 基础使用
- 在 ComfyUI 中添加 "Fast Groups Muter" 节点
- 节点会自动检测工作流中的所有组（Group）
- 每个组会显示一个开关，用于控制组内所有节点的静音状态

### 2. 属性配置
右键点击节点 -> Properties，可以配置：

- **matchColors**: 按颜色过滤组（如 "red,blue" 或 "#ff0000"）
- **matchTitle**: 按标题过滤组（支持正则表达式）
- **showNav**: 是否显示导航按钮（跳转到组位置）
- **showAllGraphs**: 是否显示所有子图中的组
- **sort**: 排序方式
  - "position": 按位置排序（默认）
  - "alphanumeric": 按字母顺序
  - "custom alphabet": 自定义字母顺序
- **customSortAlphabet**: 自定义排序字母表
- **toggleRestriction**: 切换限制
  - "default": 无限制
  - "max one": 最多一个启用
  - "always one": 始终保持一个启用

### 3. 右键菜单操作
- **Mute all**: 静音所有组
- **Enable all**: 启用所有组
- **Toggle all**: 切换所有组状态

## 实现方式

### 简化版（fast_groups_muter_entry.js）
- 独立实现，依赖最少
- 包含核心功能
- 推荐用于快速集成

### 完整版（fast_groups_muter.js）
- 完整移植 rgthree 实现
- 需要所有依赖文件
- 功能更完整，包括高级 UI 特性

## 注意事项

1. **虚拟节点**: 此节点不参与实际的工作流执行，仅用于 UI 控制
2. **组的概念**: 需要在 ComfyUI 中先创建组（Group），节点才能检测到
3. **静音 vs 旁路**: 此节点控制的是"静音"（Mute/Never），不是"旁路"（Bypass）
4. **性能**: 大量组时可能影响 UI 响应速度

## 原始来源
- 项目: rgthree-comfy
- 作者: rgthree
- 许可: 请参考 rgthree-comfy 项目的许可证

## 相关节点
- Fast Groups Bypasser: 控制组内节点的旁路状态（可类似方式移植）
- Fast Muter: 控制单个节点的静音状态
- Fast Bypasser: 控制单个节点的旁路状态
