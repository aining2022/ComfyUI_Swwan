---
name: comfyui-swwan-node-development
description: 为 ComfyUI_Swwan 添加、移植、整合或修复节点时，查询现有能力并判断复用、扩展、优化或新增，维护工作流兼容、独立注册和来源许可。
---

# ComfyUI Swwan 节点开发

先定位当前仓库；安装的 skill 可能是指向仓库的符号链接。以 `node_manifest.json` 和实际代码为准，`docs/node-catalog.json` 用于快速检索，不将技能中的例子当作固定节点清单。

用 `scripts/find_candidates.py "任务关键词" --repo /path/to/ComfyUI_Swwan` 查候选，随后只读相关接口和执行分支。目录过期时用项目的 `scripts/export_node_catalog.py` 在 ComfyUI Python 环境刷新。候选排序只帮助定位，不证明行为等价。

## 如何判断

- 同一任务已完全覆盖：复用；核心节点能完成的任务也优先复用。
- 同一任务缺少参数或执行模式：扩展现有主入口，缺省参数继续执行原算法。
- 已有输入会产生错误输出或崩溃：优化现有实现，先复现并定义错误／空输入语义。
- 任务、数据合同或运行依赖独立：新增节点，说明现有入口为什么不能承担。
- 入口被完整覆盖：保留 Legacy 包装及替代说明；不要用改菜单代替功能整合，也不要用减少注册数量作为硬指标。

比较的不只是名称和 socket 类型：核对参数默认值、输出顺序、像素取整、插值、batch/list、首帧处理、空输入、设备和 dtype。遮罩值、alpha、预乘 RGB 是不同的数据语义。

## 本项目的兼容边界

- `BOX`、逐图 `BBOX`、行列闭区间 `IMAGE_BOUNDS`、完整裁剪 `SEAM`、恢复尺寸 `STITCH3` 不可因名称相似而替换。
- ComfyUI Load Image 的 MASK 是 `1 - alpha`；RGBA Safe Pre 输出实际 alpha。保存前确认 RGB 是否预乘。
- 独立注册只在 `node_manifest.json` 维护。新 ID 用 `Swwan功能名`，显示名称带 `(Swwan)`，不添加第三方同名别名。已有独立 ID 是协议；`ImageResizeByMegapixels` 保留 ID。
- 扩展输出仅在末尾追加；新增模式保留原缺省分支。动态端口缩减不能丢失连接，隐藏控件仍需保存值。
- 重名节点的历史工作流必须确认归属；用 `scripts/migrate_workflow.py` 的显式节点范围迁移并另存，不能全局替换第三方类型。模型、提示词和采样配置只在任务要求时修改。
- 第三方实现记录来源、版本、许可和修改；整包 GPL-3.0 不替代各文件原许可。可选库在相关操作执行时加载，基础注册不依赖原插件。

实施时按功能族使用 `nodes/` 接口、`ops/` 算法和兼容包装；禁止新增 `import *`。旧模块转发和 Legacy 节点只有兼容用途，不在其中另写一套计算算法。

## 验证与交付

根据风险选择项目现有测试：注册和合同检查；旧分支回归；固定合成图片与参考像素／坐标；实际文件重读；迁移幂等和来源隔离；模式、动态端口、颜色及保存重载。测试不能依赖本机邻接插件或当前 Git HEAD。

刷新目录、替代关系和必要示例；修改公共合同必须在变更说明中列出。完整贡献规则见项目 `CONTRIBUTING.md`，硬件边界见 `docs/HARDWARE_VALIDATION.md`。CPU 通过不能写成 CUDA／RTX／模型效果通过。执行、模型下载、付费调用和发布权限沿用本次用户要求，skill 不增加授权。
