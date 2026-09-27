# 迁移与数据合同

通用迁移器支持 UI workflow（nodes/links）以及 API prompt（class_type/inputs），也支持带 `prompt` 字段的封装。默认检查后另存 `<原名>-swwan.json`，禁止覆盖输入或既有输出。`--dry-run` 只报告。

未知来源的重名节点会列为 ambiguous：必须通过 `--swwan-node-id 编号` 指定迁移范围。已有 `cnr_id=comfyui_swwan` 或 `swwan_version` 可确认本仓来源。API prompt 使用 `_meta.swwan_version`。只在文件全部为本仓历史节点时显式 `--assume-swwan`。未选择的第三方节点不修改。

重命名不改节点编号、连线、布局或旧参数；新可选控件和输出追加，不重排旧端口。需要补齐新增控件时传 `--comfyui-root`，使用实际接口。旧字体值转换为 FreeMono；Resize v2 的历史非法 keep_proportion=false 修复为 stretch（不改有效枚举值）。二次迁移结果相同。完整旧 ID 和替代关系由 [节点目录](NODE_CATALOG.md) 生成。

Qwen2511 专用迁移器会按源接口重映射裁剪 SEAM/MASK 和相关端口，补接原图／BOX；再用通用工具补齐现有新模式控件。示例是迁移结果，核心、QwenEditUtils、模型、提示词和采样不变。

专用迁移器同时保存当前输入名对应的 `widgets_values_named`，覆盖原插件的旧字段名；Swwan 模式控件在加载时按名字恢复。不同前端可能将 COLORCODE 显示为控件或仅保留插槽，不能仅凭 `widgets_values` 的长度确认兼容。验证时还需用实际前端转成 API prompt，检查转换后的枚举、数值及颜色连线，并测试保存重载。更新插件后刷新浏览器以加载新脚本。

## 不可互换的数据

| 类型 | 语义 |
| --- | --- |
| BOX | 共享裁剪矩形；本仓旧 Crop/Restore 协议 |
| BBOX | WAS 逐图边界列表 |
| IMAGE_BOUNDS | KJ 行列边界，包含其端点约定 |
| SEAM | 编辑区域原图/原遮罩/越界填充/缩放/接缝完整信息 |
| STITCH3 | Resize Sum/恢复尺寸流程的原协议 |
| IMAGE tensor batch | 同一尺寸的 NHWC tensor |
| IMAGE list | 可包含不同尺寸；不能直接当同尺寸 batch |

Legacy 保留原算法、原输出顺序和批次规则。新主入口不是每个历史合同的替换：有 replacement 的条目表示推荐选用入口，自动迁移不将 Legacy 类型强行替换为主入口。
