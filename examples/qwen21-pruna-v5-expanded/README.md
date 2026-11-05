# Pruna v5 展开版：人工审查入口

本版针对“不要把处理流程整合进一个节点”的要求重新输出。**不是 v4 聚合节点改名**：各任务的输入、mask、分支、裁剪、标记、原生编码、六步采样、解码、校色、几何恢复、最终贴回和 alpha 连接直接显示在画布上。没有 `SwwanQwen21*` 任务封装、隐藏子图、任务 config 对象或 Set/Get 虚拟连线。保留 v4 文件，不覆盖历史版本。

状态：静态图合同和 CPU 几何检查通过；**尚未完成本版的 ComfyUI 浏览器导入/保存重载及 L20 推理验收**。v4 的 GPU 图像、速度和效果结论不适用于此版本。这里交付的是供人工确认的展开候选，不是已验收的上线版本。

## 先打开哪个文件

- `pruna-v5-expanded-all.json`：总览，9 行独立任务。仅第一行 edit 启用，其余整行 muted（mode=2）。不要直接把全部行同时启用。
- `pruna-v5-edit.json`：建议先看这张，55 个可见节点；由整图/局部开关选范围。
- `pruna-v5-remove.json`：普通/增强移除，重点对照软 mask、块化和绿色不透明度。
- `pruna-v5-outpaint.json`、`pruna-v5-cutout.json`：扩图与透明抠图，各自独立。
- `pruna-v5-face_swap.json`：127 个可见节点，保留 BFS 检测、排序、六参考、逐人循环、肤色和恢复分支。
- `pruna-v5-watermark.json`、`pruna-v5-photoreal-image.json`、`pruna-v5-photoreal-text.json`、`pruna-v5-angles.json`：其余独立任务。

导入界面用不带 `-api` 的 JSON；同名 `-api.json` 仅供程序提交，不能直接用固定占位文件名请求执行。`schemas.json` 是生成/测试使用的接口快照，**不是工作流，也不是实时服务器兼容证明**。

## 导入前

1. 更新 ComfyUI_Swwan 至包含本版本的代码并重启。唯一新增节点是 `SwwanQwenGeometryUnpack`：只读取同源 Swwan SEAM，去补边并逆缩放，输出局部图和原坐标。它不做模型、校色、mask 处理或贴回；透明底、RGB 贴回、alpha 贴回仍在画布上分开。
2. 保留原工作流外部依赖：原生新版 `TextEncodeQwenImage21`、QwenEditUtils、EasyUse、原 Attention 节点、BlueBox ColorCalibrator、孤海 GH_CropRestore；BFS 另外保留原检测、排序、对齐相关插件与模型。根据缺失节点提示确认插件版本，本次未替你安装或下载。
3. 模型加载器保留原 v3 的文件名。若本机路径不同，只在对应加载器中重新选择，先不要改变 LoRA 权重/采样设置。
4. 重新选择原图和选区文件。插件允许导入时保留尚不存在的上传图片名（如 `selection.png`），不会替换成服务器上的其他图片；实际执行前仍须上传/选择有效文件。**即使 edit 处于整图模式，连接在 lazy 分支里的选区加载器仍可能被 ComfyUI 提交前校验，因此也要选择存在的文件。** 懒执行不能当成跳过文件校验的保证。
5. 独立文件推荐逐个运行。总览是审图用，同一模型加载器重复出现不代表显存复制，也不保证多个 LoRA GPU 常驻或瞬时切换。

## 三个核心问题对应到画布

### 1. 色差

| 任务 | 校色路径 | 默认值与边界 |
| --- | --- | --- |
| 编辑、写实图生图、多角度 | 干净裁剪与生成 RGB → `ColorCalibratorBlueBox` → Swwan Restore | strength=0，防止全局校色把主动改色撤销；需要保色时自行调高并对比 |
| 移除、去水印、扩图 | `GH_CropRestore` 原生校色/贴回 | 颜色还原强度=1、mask 扩展=4%、羽化=5%；最后另用原坐标权限 mask 限定可改范围 |
| BFS | 原 BFS 肤色匹配、混合与恢复分支 | 默认肤色功能关闭，开关与节点均可见 |
| 抠图 | 生成 RGBA 几何反变换 → 独立透明画布贴回 | 不从用户选择 mask 伪造 alpha，不强制全局校色 |

GH_CropRestore 本身就是原第三方“校色+恢复”节点，本版仍使用它；没有再把整个任务藏进新的封装。可对照该节点参数和生成块预览检查。

### 2. 位移/缩放

主图只经过一次 `SwwanCropByMaskV5(qwen21_aligned)`；不再串第二个 Cut BlueBox，也不在主裁剪后额外 Resize。官方编码器 `resolution=0`；EditUtils 的主图长边取已对齐图实际尺寸，随后显式去其 pad。恢复只使用对应裁剪输出的同一个 SEAM。

- Swwan SEAM 只接 Swwan 恢复/GeometryUnpack；GH SEAM 只接 GH_CropRestore，不能混接。
- “几何尺寸恢复”和“范围外像素不变”不能证明 Qwen 没有改变选区内的人脸、结构、纹理或位置。本版不承诺生成内容零位移。
- 编辑可选参考加载器默认 disabled 且不接入编码器。启用后连接可见 `images.image_2`。本版不声称它具有同场景多参考的同裁剪/零偏移能力；BFS 参考脸对齐是独立任务流程。

### 3. 尽可能减少推理像素

已有 `SwwanCropByMaskV5` 的 `qwen21_aligned` 模式包含 BlueBox 不动点画布选择，因此不应重复连接 Cut BlueBox。所有主图裁剪使用：

```text
qwen_resolution_mode = crop_area
qwen_min_resolution = 128
qwen_max_resolution = 1024
R = clamp(32 * floor(sqrt(native_crop_width * native_crop_height) / 32 + 0.5), min, max)
```

R 是**近似面积预算的边长**，不是最终图像长边，也不是原 mask 白色像素数量。先算选区外接框+上下文，再算预算，之后选 BlueBox 对齐画布；最终像素数可能在 R² 附近浮动。裁剪宽、高、实际 R、模型画布宽高都有可见数值预览。128 是下限，不会把每次任务都压成 128。小预算的模型质量和耗时仍需实测。

扩图整张扩展画布参与预算，避免只裁到狭窄边条而失去主体上下文；本版不做四边拆分推理。纯文生图使用官方编码器的默认 1024×1024 latent，不支持在此图中任意修改文生图尺寸，也不借用 SD/Flux 不兼容空 latent。

## 各任务预处理与提示词

| 任务 | 可见的专用步骤 | 任务 LoRA 权重 |
| --- | --- | --- |
| edit | 原图 → 整图/局部权限 → 单次裁剪；可选参考显式空端口 | 原 v3 附加 LoRA=0.6，可见开关默认开 |
| remove | 选区/圈选 → 扣保护 → 普通软 mask 或 32px 块化增强 → 再扣保护 → 裁剪 → 绿色标记 | 1.0；普通绿色 opacity=0.4，增强=1.0 |
| watermark | 局部/整图权限 → 裁剪 → 原触发词 → GH 恢复 | 0.8 |
| outpaint | 原图四向灰色 padding → 新区 mask → 可选 GrowMask 接缝带 → 全画布预算 | 1.0 |
| cutout | 选区/圈选 → 裁剪 → 向内腐蚀2px → 相减得红内轮廓 → EditUtils 原生编码 | 1.0 |
| photoreal-image | 整图预算 → 官方图片编码 → 写实触发词 | 0.8 |
| photoreal-text | 无图片输入 → 官方纯文本 latent → 写实触发词 | 0.8 |
| angles | 整图预算 → 官方图片编码 → 可见角度触发词 | 0.9 |
| face_swap | 原图检测/排序 → 可见逐人循环 → 六参考按 index 选择 → 脸对齐/参考预算 → EditUtils → 原 BFS 恢复 | BFS v1.1=0.7 |

所有任务先加载原 Pruna v0.2 LoRA（权重1）。保留原 INT8 DiT、INT8 CLIP、Qwen 2.1 VAE、Attention 后端、seed=482316605808414、CFG=1、Euler 和六步 sigmas：

```text
1.0, 0.961054, 0.918949, 0.769302, 0.545554, 0.222292, 0
```

固定触发词与额外指令是两个可编辑节点，拼接后的实际文本有预览。必要更正只有：angles 修正原 v3 误用的绿色移除提示词，使用 `<mva> right side view, high-angle shot`；cutout 提示词补上移除红色定位标记。没有额外加入 Cache/Spectrum 或改变六步表。

## Mask、保护、alpha 不混用

- 文件选区：已有 `SwwanLoadAndResizeImage` 关闭 resize、background_color 为空，取 **IMAGE → ImageToMask(red)**。这条路径保留灰度图的白255，不使用 LoadImage 的 `1-alpha` 作为文件选区。
- 普通主图 LoadImage 的 MASK 是 inverse alpha；它最后用于恢复原图透明度，不能当作灰度选择图。
- edit/remove/watermark/BFS 的保护默认全黑。需要时用同样的文件读取/红通道链替换全黑 SolidMask，保持原图尺寸。白色为硬保护，灰色为软值，不应声称灰色区域绝对不变。抠图不套用局部编辑保护合同；扩图用新区 mask 保护旧画布；写实/角度是整图变换。
- 圈选只使用可见 fill_holes，不做闭合检测或断口自动修补；请先看 mask 预览。普通涂抹旁路填洞以保留软值。与 v4 的严格圈选检查不同，这是展开基础节点版的明确边界。
- 移除的增强块化在原图坐标执行，之后再次扣保护；不要把绿色标记图当干净校色参考。
- 最终门控取有效 mask 的非零支持域；软融合在恢复阶段完成，门控只是阻止范围外泄漏。
- 扩图 RGB 与 inverse alpha 的四向像素值由同四个 PrimitiveInt 驱动；新增区域不透明。GrowMask=0 保护整个旧画布，正值明确允许改动旧图接缝带。仅支持像素数，不支持百分比。
- 抠图 `require_alpha=true`：VAE 若没有输出真实 RGBA 则停止报错；有 RGBA 后，RGB 和 inverse alpha 分开贴回原尺寸透明画布，不用用户选区裁掉模型补全部分。
- BFS 主图 LoadImage MASK 保留原工作流“mask editor 画选区/alpha”的使用方式，不会自动从 RGB 图读出选择区。检测排序固定在原图，循环裁剪/恢复使用累计结果；不宣称与 v4 新循环逻辑完全等价。六个参考加载器均为可见有效节点，即使只换1人，也可能需要为未使用槽位选择存在的文件以通过预校验。

## 建议手动验收顺序

1. 先只导入 edit：确认没有红色缺失节点、控件对应正确、原模型/LoRA/六步没变，选好文件，再保存重载检查。
2. remove：同图同 seed 比较普通/增强。检查绿色图、裁剪 mask、原坐标权限与最终未选区域。
3. outpaint：先单边扩图、GrowMask=0，核对输出尺寸和旧画布像素；再检查非零接缝带。
4. cutout：先看原 VAE 解码是否真实含 alpha，再看透明贴回是否保留补全部分。
5. BFS：先1人1参考，再多人；核对原图排序、参考 index、累计结果和保护，最后才比较肤色开关。
6. 对相同输入比较 R=128下限/512下限/1024上限的实际画布、耗时、显存与质量；不能仅凭预算更小断言质量可用。

## 可复现检查

从插件根目录运行：

```sh
python tests/test_pruna_expanded.py
python tests/test_pruna_expanded_bfs.py
```

测试读取仓库内工作流、schema快照和已保存 v3 baseline，不依赖 Downloads 或邻接插件。完整 BFS 重建需要原始用户 BFS UI 文件（来源哈希记录于 provenance.json）；不将用户下载目录当测试前提。
