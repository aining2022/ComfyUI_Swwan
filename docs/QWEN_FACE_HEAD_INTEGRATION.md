# Qwen 换脸／换头工作流的纯图片处理整合

迁移副本：[`examples/qwen-face-head-swwan.json`](../examples/qwen-face-head-swwan.json)。原文件保持不变。201 个节点中替换了 43 个实例、23 种图片／遮罩节点；节点编号、布局、启用／绕过状态和处理顺序不变，还原节点额外补接原图与 BOX 两条连线。

## 节点取舍

| 原能力 | Swwan 入口 | 处理 |
| --- | --- | --- |
| Mask Fill Holes、ToBinaryMask、MaskFix+、LayerMask: MaskGrow、GrowMaskWithBlur | `SwwanMaskProcess` | 新增一个清理入口，分别保留填孔、阈值、清理和两种扩展算法，按模式显示参数 |
| 孤海-遮罩混合运算 | `SwwanMaskCombine` | 新增：8 种运算、BBOX 取景和对齐；有效区域尺寸不能用画布尺寸替代 |
| 孤海遮罩分析、GuHaiMaskDetect | `SwwanMaskAnalyze` | 新增一个分析入口，前 7 个输出为原位置／尺寸，末尾为面积检测 BOOLEAN |
| MaskToSEGS、SegsToCombinedMask、ImpactSEGSToMaskBatch、孤海Seg次序过滤 | `SwwanMaskSegments` | 新增一个区域入口，保留七字段 SEGS、量化、排序／分组和索引规则；无检测模型依赖 |
| RemoveBackgroundWithMask | `SwwanImageMatte` | 新增：使用已有 MASK 抠图，输出直通 RGBA、背景填色 RGB 和 MASK；与预乘 RGBA Safe Pre 语义不同 |
| GH_MaskCropV2 | `SwwanCropByMaskV5` | 扩展编辑区域模式，支持直接输入原系数；连线系数也按短边扩展，不误当成 reserve_ratio |
| ImageResize+、图像缩放V2_孤海 | `ImageResizeKJv2Alternative` | 扩展 `essentials` 原算法模式，复用既有编辑尺寸模式；保留插值、取整、条件执行和填色 |
| ImageColorMatch+ | `SwwanColorMatch` | 扩展 `mean_std` 模式，支持六种色彩空间、参考遮罩、设备及分批；旧 color-matcher 模式保持原值 |
| ImageMaskPreview_Guhai | `SwwanImageAndMaskPreview` | 扩展 `regions` 模式，保留首张图叠加多个遮罩和序号；原 KJ 预览保持默认 |
| GH_CropRestore、GulfSeaImageMergeMask、图像缩放范围孤海、BlockifyMask、DrawMaskOnImage | 已有 Restore、Blend、Range、Blockify、Draw 入口 | 复用已有实现，重映射 SEAM／原生 MASK 输出及端口 |

新增 5 个任务入口全部位于 `Swwan/Advanced/Mask` 或 `Swwan/Advanced/Image`。21 个推荐主入口保持不变；总注册数为 118（77 个 Advanced、18 个 Legacy、2 个 Experimental）。所有新增 ID 独立，不注册原插件的冲突别名。

## 数据与兼容

- WAS 填孔先通过 PIL 量化，再按非零区域填充；保留原单张输出 `[1,1,H,W]`，扩展成可逐图处理的 `[B,1,H,W]` 批次。Binary 模式按 Impact 的 `threshold / 255` 比较，输出 `[B,H,W]`。Cleanup 的 `close_holes` 是灰度 closing，不等同于二值填孔。
- LayerStyle 的扩展与 KJ 的扩展不能互换：前者保留 PIL 量化／模糊，后者保留角点、逐帧增量、lerp 和 decay。
- 遮罩分析使用首张和 `>=0.5`，空遮罩的边界返回整个画布；面积检测使用整个输入批次和 `>0.5`。两套边界语义明确保留。
- SEGS 转合并／批次 MASK 保留 uint8 量化；排序／筛选保留软遮罩及原裁剪坐标。正数筛选数量使用循环索引，数量 0 表示从起始索引到末尾；空 SEGS 批次返回一张黑遮罩。
- 原字体 `苹方特粗.ttf` 不随包分发，迁移为已有、可分发的 FreeMono。开启序号时字体外观会改变；叠加、字号与坐标规则保留。运行时也接受旧值并明确转到 FreeMono。
- 裁剪的新增系数输入位于末尾，默认 `reserve_ratio` 仍走原计算；仅迁移的编辑区域节点选择 `factor`。已有 BOX 和输出顺序不变。

## 范围与迁移

按本次用户确认，RMBG、人体分割、SAM3、DW 姿态检测、人脸对齐等模型预处理节点保留。QwenEditUtils、核心模型／VAE／采样节点、提示词、模型文件名、循环、Set/Get、数值／布尔控制、缓存预览桥和批次进度控件保留。`LG_Color_Match_V2` 按用户要求忽略，节点及参数完全不变。因此完整工作流仍需这些原插件和模型，不能把本次纯图片处理迁移当作整个工作流无第三方依赖。

```sh
python scripts/migrate_qwen_face_head_workflow.py source.json new-swwan.json --comfyui-root /path/to/ComfyUI
```

这是该工作流的显式迁移入口；只选取表中纯处理类型，拒绝覆盖原文件和已有目标。重复迁移不会继续更改节点、参数或连线。后端算法不导入原第三方插件；必要的 scipy/OpenCV/torchvision/kornia 在执行对应功能时加载。kornia 属于 ComfyUI 依赖，color-matcher 仅旧颜色模式需要。

## 验证证据

`tests/test_face_head_processing.py` 的 7 项测试覆盖多组固定合成输入：所有遮罩混合／BBOX／对齐选项，填孔／二值／形态学／模糊，五种区域排序、两种方向、循环索引及空区域，抠图／描边，序号预览，全部缩放条件与方法，以及六种颜色空间和参考 MASK。参考源码冻结在 `tests/fixtures/face_processing/`，`sources.json` 记录来源和版本，不依赖 Downloads 或邻接插件。

同时检查迁移的类型、保存参数、必需端口、连线、幂等及未选中节点的保持；通过模拟“裁剪 → 清理 → 区域筛选 → 抠图 → 颜色匹配 → 合成 → 还原”验证位置、原图尺寸和输出遮罩。旧合同／Qwen2511 回归、可选库缺席注册及前端控件保存另由仓库现有测试覆盖。

以上为 CPU 图片处理验收。未下载模型，未执行人脸检测／分割、Qwen 推理或 CUDA 路径，不代表换脸效果或完整生成链已验证。

实际前端脚本 `tests/frontend_face_head_browser.js` 在只加载 Swwan 和 ComfyUI 核心图片／遮罩节点的临时 CPU 服务上创建 15 节点、17 连线的处理图，确认模式显示、外接 FLOAT 系数、COLORCODE、SEGS／BOOLEAN 端口及保存重载。通过前端转成 API 后实际排队执行，重读输出为 128×96 RGB PNG，证据为 `tests/fixtures/face-head-browser-acceptance.json`；该图没有模型或第三方节点。


## 已编辑版本的修复精简验收

新增 [`qwen-face-head-final-swwan.json`](../examples/qwen-face-head-final-swwan.json) 是用户后续修改的 179 节点版本的精简副本，不重新生成自最初的 201 节点文件。保留原文件及用户已有的模型、提示词、控件、布局与旁路状态；最终 145 个节点、167 条连线，唯一最终输出为 SaveImage #163（`换脸_无高清`）。

精简仅删除最终保存链不可达的预览／序号／进度、辅助链、面板及闲置模型配置。Set/Get 虚拟连接已解析，全部可切换分支保留；循环 initial_value1、索引和计数仍有效，展示输入 initial_value2/3 清空但端口未重排。原旁路缓存的图像消费者改接原贴回图像。

修复 Resize #92 的错位参数、Draw #62 的旧接口迁移和 Matte #47/#58 的白色背景值。统一前端按名称保存参数，并在 COLORCODE 控件缺席时保留其合法字面颜色；外接颜色优先。详见 [迁移说明](MIGRATION.md)。

- `tests/test_face_head_workflow_repair.py`：当前基准、145/167、全部切换分支、布局和参数保持、必填端口、严格坏值报错、虚拟变量、幂等、旧版 Draw 两／三参数、白色绘制、原 Essentials 像素与接缝还原。
- `tests/test_face_head_loop.py`：冻结的 EasyUse 图展开实现，3 次模拟裁剪编辑／贴回，移除展示状态后的累计结果由真实 SaveImage 保存并重读；不是普通 Python for 循环替代验收。
- `tests/frontend_face_repair_browser.js`：隔离 CPU 服务中的 11 节点／12 连线图，分别在颜色控件存在／模块缺席时运行，禁用 LiteGraph 全局命名恢复以检查本仓逻辑，验证外接宽度和颜色、保存重载及相同执行参数。真实排队输出白色背景 RGB、外接 #123456 背景 RGB 和 768×576 缩放 PNG，证据为 `tests/fixtures/face-repair-browser-acceptance.json`。

完整多人换脸生成仍依赖原模型预处理、EasyUse 等插件及模型；这些模型未运行，CPU 验收不代表最终换脸效果或远程生成通过。
