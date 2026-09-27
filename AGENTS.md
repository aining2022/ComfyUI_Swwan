# ComfyUI_Swwan：AI agent 工作入口

本文件面向操作仓库的 AI agent，适用于整个仓库。使用者入口是 [README.md](README.md)；节点开发的专项流程是 [SKILL.md](skills/comfyui-swwan-node-development/SKILL.md)。遵循本次用户要求的范围和授权，默认用中文报告结论、验证与未解决项。

## 先定位事实

确认实际仓库和 `git status --short`，保留用户未提交改动。不要假定聊天 cwd、本机插件目录或已安装 skill 就是本次目标。先读与任务有关的文件，不做无关全量整改。

| 信息 | 权威入口 | 使用方式 |
| --- | --- | --- |
| 注册 ID、类、菜单、层级、来源、许可、旧 ID、替代关系 | [node_manifest.json](node_manifest.json) | 唯一注册清单；[registry.py](registry.py) 拒绝内部重复 ID |
| 实际输入、默认值、输出、list 标志和算法 | manifest 指向的 Python 类／执行分支 | 最终依据；名称和类型相同不证明行为相同 |
| 接口的机器可读索引 | [docs/node-catalog.json](docs/node-catalog.json) | 查询候选；与实际实现不符时重新导出 |
| 人类节点导航 | [docs/NODE_CATALOG.md](docs/NODE_CATALOG.md) | 生成文件，不手工维护第二份清单 |
| 开发决策和贡献规则 | [节点开发 skill](skills/comfyui-swwan-node-development/SKILL.md)、[CONTRIBUTING.md](CONTRIBUTING.md) | 比较复用、扩展、优化和新增 |
| 迁移／透明度／许可 | [MIGRATION](docs/MIGRATION.md)、[RGBA](docs/RGBA.md)、[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) | 修改相应合同前读取 |
| 验证证据 | [PROJECT_REVIEW](docs/PROJECT_REVIEW.md)、[Qwen 换脸／换头](docs/QWEN_FACE_HEAD_INTEGRATION.md)、[硬件清单](docs/HARDWARE_VALIDATION.md) | 区分本地 CPU、前端、GPU 与完整推理证据 |

不要在本文件复制全部节点或固定数量。需要当前数量时从 manifest 按 `tier` 统计；接口目录通过实际注册导出。

## 修改前的判断

从仓库根目录查询候选，无需启动 ComfyUI：

```sh
python skills/comfyui-swwan-node-development/scripts/find_candidates.py "缩放" --repo . --limit 8
```

随后读取相关接口和算法，比较任务、输出语义／顺序、默认值、batch/list、广播／首帧／空输入、MASK／alpha／预乘、像素取整、插值、设备／dtype 和依赖。

| 结论 | 行动 |
| --- | --- |
| 现有入口或核心节点完整覆盖 | 复用，补必要连接／说明 |
| 同任务缺少选项或模式 | 扩展已有入口，缺省仍执行旧算法 |
| 既有合法输入出现错误 | 优化现有实现，固定可复现行为并针对性验证 |
| 任务、数据合同或运行依赖独立 | 新增；写明现有入口不能承担的原因 |

候选排序仅用于发现。同名、同为 IMAGE 输出或菜单更少都不能证明整合正确。语义独立的基础工具保持专用入口，Legacy 保留历史合同。

## 实施合同

- **结构**：`nodes/` 提供接口，`ops/` 放算法／共享辅助函数；根模块保留必要的明确转发或兼容包装。禁止新增 `import *`，不在包装中复制另一套算法。
- **注册**：只改 manifest 注册。新 ID 使用 `Swwan功能名`，显示名带 `(Swwan)`；既有不冲突 ID 保留，不添加第三方冲突别名。
- **菜单**：主入口按 Image／Mask／Batch／IO／Utils；既有 RGBA 专用分组保留。专用能力 `Advanced/<任务>`，历史包装 `Legacy`，设备实验功能 `Experimental`。
- **兼容**：原必填输入、默认值、输出前缀顺序、list 标志和缺省算法保持。新模式可选，输出只在末尾追加；必要的合同变更逐项记录在 CHANGELOG 和迁移说明。
- **数据**：BOX、BBOX、IMAGE_BOUNDS、SEAM、STITCH3 不可互换；tensor batch 和 list 分开处理。Load Image 的 MASK 是 `1-alpha`，RGBA Safe Pre 输出实际 alpha。
- **裁剪**：`edit_region` 保留首张行为，SEAM 包含恢复信息；原图与 BOX 仍满足 Restore 的必填接口。外接原扩展系数使用 `factor` 分支，不能误当成 reserve ratio。
- **前端**：使用本仓 `web/js`。隐藏参数仍保存，转换为输入的控件遵循 ComfyUI 行为；缩减动态端口不丢失已连接端口，保存重载后值和连线保持。
- **依赖**：原第三方插件不作为处理算法运行依赖。可选 vision／color／GPU 库在相关执行处加载；基础注册不加载模型和 GPU 后端。沿用现有 ComfyUI 的 torch，GPU 库按架构配置。
- **来源**：移植代码记录来源、版本、版权、许可和修改，更新 manifest、文件声明与 THIRD_PARTY_NOTICES；冻结测试参考也记录来源。整包 GPL-3.0 不替代原 MIT 等声明，不分发来源／许可不明资产。

常规实现细节在授权范围内自行处理。不要覆盖用户改动、重置工作树或扩大任务；提交、推送、发布、模型下载、推理及付费操作按本次授权执行。

## 工作流迁移

[通用工具](scripts/migrate_workflow.py) 支持 UI workflow 与 API prompt。先检查，再另存；来源不明的重名节点必须选择归属，不能全局替换第三方 ID。

```sh
python scripts/migrate_workflow.py input.json --dry-run
python scripts/migrate_workflow.py input.json output-swwan.json --swwan-node-id 12 --comfyui-root /path/to/ComfyUI
```

`12` 替换为实际属于 Swwan 的编号。只有用户范围和来源确认整个文件属于本仓历史节点时，才使用 `--assume-swwan`。工具拒绝覆盖原文件与已有输出。

Qwen 示例使用各自的显式迁移器，不把任意工作流套入特定映射：

```sh
python scripts/migrate_qwen2511_workflow.py input.json output-swwan.json --comfyui-root /path/to/ComfyUI
python scripts/migrate_qwen_face_head_workflow.py input.json output-swwan.json --comfyui-root /path/to/ComfyUI
```

检查节点编号、布局、端口、参数和连线；允许的补接明确记录；重复迁移幂等，未选中的第三方节点不变。保留模型文件名、提示词、采样、QwenEditUtils 与模型预处理，除非本次用户明确要求修改。现有换脸／换头示例中的 `LG_Color_Match_V2` 未迁移，不能宣称其已由本仓替代。

## 验证命令与边界

在 ComfyUI Python 环境中，从仓库根目录运行下列命令，替换 `/path/to/ComfyUI`。Python 最低要求见 [pyproject.toml](pyproject.toml)；CPU CI 的 ComfyUI 固定版本与 Python／Node 配置以 [.github/workflows/cpu.yml](.github/workflows/cpu.yml) 为准，该参考版本不等于最低支持版本。

按改动选择相关检查，不因文档修改执行图片／模型测试：

| 改动 | 必要证据 |
| --- | --- |
| 注册、接口、依赖 | manifest 无重复、旧合同回归、缺少可选库仍注册 |
| 算法、批次、空值、alpha | 固定合成输入；尺寸／坐标／像素和边缘情况；旧缺省分支回归 |
| 保存 | 临时目录实际写入并重读像素、alpha、元数据、路径、编号和 sidecar |
| 前端 | 对应 Node 检查；控件或端口变化再验证实际浏览器的连接和保存重载 |
| 迁移 | 显式来源范围、幂等、原文件和未选中节点保持 |
| 分发／文档 | 文件与链接存在；新增分发资产检查 wheel 内容；生成目录与 manifest 一致 |
| GPU／模型 | 独立硬件或实际推理证据，不从 CPU 结果推断 |

```sh
# CPU 合同与图片参考
python tests/test_project_contracts.py --comfyui-root /path/to/ComfyUI
python tests/test_optional_imports.py --comfyui-root /path/to/ComfyUI
python tests/test_workflow_image_nodes.py --comfyui-root /path/to/ComfyUI
python tests/test_face_head_processing.py --comfyui-root /path/to/ComfyUI

# 前端逻辑与 skill 候选
node tests/test_workflow_image_modes.mjs
node tests/test_project_frontend.mjs
python tests/test_skill_candidates.py

# 注册／接口变化后重新生成两个节点目录
python scripts/export_node_catalog.py --comfyui-root /path/to/ComfyUI
```

共装检查仅在原插件和依赖已具备时执行；它会运行第三方插件导入钩子：

```sh
python scripts/check_coinstall.py --comfyui-root /path/to/ComfyUI --plugin /path/to/other-plugin
python scripts/check_coinstall.py --comfyui-root /path/to/ComfyUI --plugin /path/to/other-plugin --reverse
```

分发检查：

```sh
python -m pip wheel . --no-deps --wheel-dir /tmp/swwan-dist
python tests/test_distribution.py /tmp/swwan-dist/comfyui_swwan-1.0.0-py3-none-any.whl
```

文件名中的版本按 pyproject 的实际值替换。构建产物不提交；保留可能原已存在的用户产物。

参考实现随 `tests/fixtures` 提供，不依赖作者的 Downloads、邻接插件路径或当前 Git HEAD。实际前端验收入口为 [frontend_browser.js](tests/frontend_browser.js) 和 [frontend_face_head_browser.js](tests/frontend_face_head_browser.js)，在隔离服务执行并保留原有用户会话；它们不是直接运行即可启动浏览器的 CLI。步骤和证据见对应验收报告。

必要检查通过后停止；只有新修改、失败或未解决的问题需要扩大／重跑。本机无对应硬件或未获推理授权时，明确报告待验项，不下载模型凑验收。

## 完成前

1. 确认行为满足本次请求，旧接口和默认分支保持；兼容变更或缺陷修复已说明。
2. 注册／接口变化后导出目录；按实际改动更新 README、相关说明、示例、CHANGELOG 和来源声明，避免重复维护数量。
3. 检查 `git diff --check` 和实际修改范围，用户原有改动仍在；提交时仅选择授权路径／内容。
4. 报告改动结果、执行过的验证和仍未验的硬件／推理边界；提交／推送状态据实际结果说明。
