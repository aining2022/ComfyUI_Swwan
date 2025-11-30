# Fast Groups Muter Web 文件说明

## 文件用途

### 入口文件
- **fast_groups_muter_entry.js** - 简化版入口（推荐）
  - 独立实现，最少依赖
  - 包含核心功能
  - 适合快速使用

- **fast_groups_muter.js** - 完整版（来自 rgthree）
  - 完整功能
  - 需要所有依赖文件
  - 更丰富的 UI 特性

### 依赖文件（从 rgthree-comfy 复制）
- **base_node.js** - 基础节点类
- **constants.js** - 常量定义
- **rgthree.js** - rgthree 核心功能
- **utils.js** - 通用工具函数
- **utils_canvas.js** - Canvas 绘制工具
- **utils_widgets.js** - Widget 工具类
- **feature_import_individual_nodes.js** - 节点导入功能

### 子目录
- **common/** - 共享组件
  - dialog.js - 对话框
  - shared_utils.js - 共享工具函数
  
- **services/** - 服务模块
  - fast_groups_service.js - 组管理服务
  - key_events_services.js - 键盘事件服务

## 使用建议

如果只需要基础功能，使用 `fast_groups_muter_entry.js` 即可。

如果需要完整的 rgthree 体验（包括高级 UI、帮助对话框等），使用 `fast_groups_muter.js` 及其依赖。

## 路径说明

这些文件会被 ComfyUI 自动加载，因为在 `__init__.py` 中设置了：
```python
WEB_DIRECTORY = "./web/js"
```

ComfyUI 会将这些文件映射到：
```
/extensions/ComfyUI_Swwan/fast_groups_muter_entry.js
```
