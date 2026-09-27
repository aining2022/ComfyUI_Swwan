# 贡献与节点开发

新增功能先查询唯一注册清单 `node_manifest.json`、生成的 `docs/node-catalog.json` 和相关实现。可使用 `skills/comfyui-swwan-node-development/scripts/find_candidates.py`；名称不同或同为 IMAGE 输出并不证明任务不同。

## 决策顺序

| 情况 | 处理 |
| --- | --- |
| 现有入口完整覆盖任务和合同 | 复用 |
| 同一任务缺少选项／模式 | 扩展主入口，保持缺省旧分支 |
| 已有合法输入产生错误 | 优化原实现，先固定可复现合同 |
| 数据合同、任务或运行依赖独立 | 新增，并写明现有节点不能承担的原因 |

比较输出语义、顺序、list 标志、batch 广播／首帧／空值、MASK 和 alpha、预乘、像素取整、插值、设备和依赖。BOX/BBOX/IMAGE_BOUNDS/SEAM/STITCH3 不可互换。核心节点能够直接完成的任务无需再复制入口。

## 代码与注册

- `nodes/` 按任务提供节点接口；`ops/` 放算法和共享辅助函数。旧根模块保留明确名称转发或 Legacy 包装，避免新增一套重复计算。
- `node_manifest.json` 是唯一注册入口：ID、实现、显示名、分类、层级、来源、许可、旧 ID 与替代关系。内部重复 ID 会阻止注册。保留既有独立 ID，新 ID 用 `Swwan功能名`，显示名带 `(Swwan)`，禁止第三方冲突别名。
- 主菜单为 Image/Mask/Batch/IO/Utils；专用工具 `Advanced/<任务>`，旧入口 `Legacy`，设备功能 `Experimental`。不能因功能名字相似就把语义独立的工具归旧版。
- 原必填输入、参数默认值及输出前缀保持；新增模式可选、输出末尾追加。动态输入缩减不能删除已连接端口，隐藏控件继续保存。
- 禁止 `import *`。模型、GPU 和可选图片库在执行相关功能处加载，并提供具体缺失依赖错误。
- 保留来源文件许可、版权与修改说明，更新 THIRD_PARTY_NOTICES 和 manifest。整包 GPL 不替代 MIT 原声明。不能捆绑来源或许可不明确的字体／资产。

## 验证

按 [AGENTS.md](AGENTS.md) 的改动类型选择 CPU、合同、前端和 skill 检查；冻结参考在 `tests/fixtures`，不能依赖作者 Downloads、邻接插件、当前 Git HEAD。行为变化对比固定合成输入、尺寸、像素、空值、alpha 和批次；保存必须重读实际文件。工作流迁移验证幂等、显式来源范围、旧输出端口和参数。

新增／扩展后刷新 `scripts/export_node_catalog.py`，补充替代关系、必要无模型示例和 CHANGELOG。GPU 只能按 HARDWARE_VALIDATION 的独立证据验收，不能从 CPU 推断。保留用户未提交改动；提交、推送、发布和付费调用遵循用户授权。

## Skill 场景验收

| 请求示例 | 结论和依据 |
| --- | --- |
| 再新增一个支持固定宽高的 Resize | 查询 Resize Image 和 MP/Range 合同；同任务被覆盖时复用。仅新增尺寸规则时扩展模式，禁止增加重复入口。 |
| 修复空列表或单遮罩批次崩溃 | 优化现有 list/crop 节点；先确定空值和广播语义，保持输出数量并加入复现测试。 |
| 新增网格块化遮罩 | MASK→MASK、块大小、块统计属于独立遮罩任务；Crop 的 SEAM/BOX 合同不能代替，允许独立主工具。 |
| 移植第三方缩放／裁剪 | 核对原接口与像素取整；同任务扩展模式，合同独立才新增。记录许可/来源/修改，用独立 ID 和来源限定迁移，旧输出追加方式兼容。 |

`tests/test_skill_candidates.py` 检查四个场景的候选和合同依据；它不是自动语义判定器。Skill 格式另用 Codex skill-creator 的 `quick_validate.py` 检查。可把本目录 skill 符号链接到 `~/.codex/skills/comfyui-swwan-node-development`，仓库仍是唯一维护源。
