# ComfyUI_Swwan 1.0.0 整改验收报告

日期：2026-09-26。P0–P2 本地实施完成；未提交、推送或发布。完整节点能力保留，选择入口收敛到 21 个主入口。硬件和远程 CI 边界单独列出。

## 结果

| 项目 | 整改前工作树 | 当前 |
| --- | --- | --- |
| 注册能力 | 112 | 113（新增统一保存主入口） |
| 选用层级 | 无统一层级 | 21 primary、72 Advanced、18 Legacy、2 Experimental |
| 分类 | 25 | 15；按 Image/Mask/Batch/IO/Utils、Advanced 任务、Legacy、Experimental 分类 |
| 跨插件冲突 | 56 KJ、4 LayerStyle，Seed 使用 rgthree ID | 61 个独立改名，无冲突别名；三插件实际加载两种顺序无 Swwan 重叠 |
| 注册来源 | 多个模块映射表 | 唯一 node_manifest，重复 ID 拒绝注册；删除过期模块映射 |
| 前端 | 失效 rgthree 链、缺失相对模块及无后端脚本 | 5 个本仓脚本；动态输入、种子、颜色、模式、FastPreview |
| 大文件 | image_nodes 4469 行、main_unit 2686 行、C_math 约 1024 行 | 原模块为明确名称兼容转发；节点按 image/math/scheduling/list 等功能族拆分 |
| 依赖 | 基础导入耦合图片／GPU 可选库 | 基础注册不加载被测可选库；相关执行处按需加载并报具体错误 |
| 开源交付 | 缺元数据、来源集中声明和维护判断依据 | 1.0.0 pyproject、GPL3 整包、原许可、贡献指南、固定参考 CI、仓库 skill |

当前 ID、菜单、来源、接口、list 标志和替代关系由 [NODE_CATALOG.md](NODE_CATALOG.md) / [node-catalog.json](node-catalog.json) 生成。README 是唯一选用总入口。

## P0 完成证据

| 条目 | 实现与验证 |
| --- | --- |
| 独立注册、命名 | registry.py / node_manifest；113 实际类与 INPUT_TYPES 全部成功；21 主入口；56+4+1 改名清单；保留 ImageResizeByMegapixels、用户 SwwanDrawMaskOnImage 改动 |
| Resize v2 默认值 | 枚举中原 default=false 无法通过 ComfyUI 验证；修正为 stretch，迁移修复同一非法字面值；有效选项和计算分支不变 |
| 通用迁移 | UI/API、显式来源范围、默认另存、拒绝覆盖、追加控件／输出、字体迁移；未选择第三方不变，重复迁移相同 |
| Bounded Crop | 单遮罩广播、等数量逐图、不等数量复用首张；合成输入直接检查两帧裁剪像素和位置 |
| Mask Crop | 空遮罩返回对应原图；不可组成同尺寸 tensor 时明确 ValueError；原非空分支保留 |
| Mask Crop Batch | 保持跳过空遮罩；全空返回 0×H×W×3 IMAGE 和 0×H×W MASK 两个输出 |
| List to Batch | 空列表明确 ValueError，不制造未知尺寸图片 |
| Seed | 独立按钮／解析；移除冲突的 Comfy 自动生成后控制；随机、递增、递减、上次、保存重载；CPU 实际输出种子等于 prompt/workflow 元数据；API 特殊种子随机回退保持 |
| 动态输入 | 5 种多输入节点独立前端；端口随 inputcount 更新；缩减已连接端口时保留连线及有效数量；实际保存重载通过 |
| 基础可选依赖 | 阻止 cv2/scipy/skimage/matplotlib/spandrel/color_matcher/triton/sageattention/nvvfx 导入后仍完成注册和全部 schema；无原插件运行时导入 |

实际共装加载：KJ 249、LayerStyle 171、rgthree 24 个本机可注册节点；Swwan 113。正反两种顺序均保持所有 Swwan 类的注册归属，并且 Swwan 与三个插件交集为空。第三方可选 Triton 和 guidedFilter 报告缺失，本次不声明其功能可用。记录见 `tests/fixtures/coinstall-{forward,reverse}.json`；可用 `scripts/check_coinstall.py` 在其他环境复验。共装只验证注册，未执行第三方模型或设备功能。

## P1 完成证据

| 功能 | 当前行为与验证 |
| --- | --- |
| 保存 | SwwanSaveImage 统一 5 格式、alpha、预乘、路径、目录独立编号、元数据、caption/JSON；共享编码／metadata／历史路径辅助；原保存合同保留。实际重读 PNG/WebP/TIFF/BMP 无损像素，JPEG 允许有损误差，alpha/尺寸/元数据/sidecar/绝对路径正确；WebP 新入口启用 exact 保留透明像素 RGB。 |
| 网格／拼接 | 条带旧分支保留；多输入 grid 和 batch_grid；固定网格包装共享拼接算法。新旧同尺寸像素相同，批次网格验证通过。 |
| 数学 | AST Expression 追加预设与末尾 BOOLEAN，原 INT/FLOAT 顺序不变；Calculate 保留 FLOAT/INT/BOOLEAN 及错误零值，所有合法预设与冻结原实现对照。无任意 Python eval。 |
| 比例缩放 | Resize Image aspect_ratio 模式共享原算法；Legacy 保留 mask-only 和 IMAGE/MASK/BOX/INT/INT。比例／fit／6 插值遍历像素对照一致；不改变 BOX 定义。 |
| Qwen 整合 | Crop/Restore SEAM 成对、MASK 末尾追加；原尺寸规则、遮罩合成、范围／块化／RGB／颜色保留参考结果。模拟编辑链返回原图尺寸；不调用 Qwen。 |

仅保留有实质任务／数据合同差异的专用工具。Add、Blend、时序过渡、通道、调色和模型补丁继续独立，Color Shift Fix 属于专用工具。旧保存路径的唯一已确认修复：SaveImageKJ 自定义相对目录从错误 cwd 路径改为 Comfy output 子目录；旧 IO 保存 wrapper 的正常 cwd 相对路径合同不改。

## P2 完成证据

- `nodes/` 为按任务拆分的接口，`ops/` 为算法／共享辅助；原 Python 模块保留明确转发。生产代码无 `import *`。裁剪/还原、比例、固定网格、编码和数学允许函数复用，不合并不兼容的数据语义。
- `pyproject.toml` 1.0.0 与可选依赖分层、CHANGELOG、CONTRIBUTING、迁移说明、硬件清单和 README 总入口。wheel 本地构建成功，检查资产／许可与前端文件齐全，TTNorms 和 .DS_Store 不在分发包。
- 全部实现来源和历史 revision 的已知／未知范围明确记录在 THIRD_PARTY_NOTICES；保留 GPL/MIT/原版权。两款 FreeMono 字体有嵌入的 GNU FreeFont 许可与 embedding exception。TTNorms 移出包，历史枚举值明确映射到 FreeMono。
- 仓库 `skills/comfyui-swwan-node-development/SKILL.md` 是维护源，已符号链接安装为本机同名 Codex skill。查询工具直接读取生成目录；四种场景（重复缩放、批次缺陷、独立遮罩、第三方移植）检查通过，skill-creator 格式校验通过。判断过程及数据合同见 CONTRIBUTING。

## 验证记录与限制

本机：macOS、Python 3.13.11、torch 2.11 开发版、CPU；ComfyUI `4e024cb1e8423be735b919c585669dc189735c7b`。

| 验证 | 结果 |
| --- | --- |
| Qwen/旧分支参考 tests/test_workflow_image_nodes.py | 7 项通过；大量尺寸、模式、遮罩与像素子案例；原文件参考随仓库提供 |
| tests/test_project_contracts.py | 9 项通过；112 旧 schema/输出前缀/list 合同、4 问题、网格、数学、比例、真实保存、迁移、Seed、两份示例实际执行 |
| tests/test_optional_imports.py | 113 注册及全部 schema 在被测可选库不可导入时通过 |
| 两套 JS harness | 模式、连接选择、隐藏值、颜色、种子、动态端口通过；原孤海颜色脚本同时加载的两个顺序也通过 |
| 真实 Comfy 浏览器 | 隔离端口与临时目录，5 本仓扩展无模块缺失；追加输出、模式、COLORCODE 连接和值、动态端口、Seed 实际执行／元数据、保存重载通过。记录 browser-acceptance.json；浏览器脚本可复验。两份 CPU 示例也从实际 UI workflow 生成 API prompt 执行成功，/view 可读取网格图与透明 PNG，记录 browser-examples.json。 |
| Skill | 4 场景与格式校验通过，本机安装路径指向仓库 |
| 静态/交付 | Python/JS 语法、git diff --check、原 Downloads 文件/.DS_Store、核心/Qwen 参数与布局、目录和 wheel 检查通过 |

允许变化逐项记录在 CHANGELOG：61 ID 改名、Resize v2 非法默认值修复、可选模式/控件、末尾追加输出、4 缺陷、已确认路径错误、拒绝任意 Python、字体 fallback 与菜单/显示。其余 schema/默认值/list 标志和被测原分支保留。合同检查不等于所有 113 种硬件／视频／模型功能都做过运行验证。

CPU CI 已配置固定 ComfyUI SHA，未运行远程 GitHub Actions（未提交/推送）。CUDA、RTX、SageAttention、MiniMax H3、屏幕／摄像头和 Qwen 完整生成效果仍按 HARDWARE_VALIDATION 独立验收；本轮没有模型下载或推理。

用户既有 `.DS_Store` 与 DrawMask ID 修改保留；Downloads 原文件保留。Qwen 核心、QwenEditUtils、模型名、提示词、采样和布局保持原值；仅为还原补接原图时增加 LoadImage 输出连线。本地整改未自动提交、推送或发布。

## 后续：Qwen 换脸／换头图片处理

本次扩展新增 5 个 Advanced 工具，当前实际注册为 118 个，推荐主入口仍为 21 个。新增图片行为、工作流迁移、参考来源和验收边界见 [专项报告](QWEN_FACE_HEAD_INTEGRATION.md)；原 1.0.0 整改基线记录保持原样。
