# Qwen Image 2.1 Pruna v4：多任务可展开工作流

本工作流以用户的 `pruna_q_image_2.1_v0.2_6step_image_edit-v3.json` 为基础，保留 Qwen Image 2.1 INT8、Pruna v0.2 upsampled LoRA、Euler、CFG=1 和原六步 sigma；加入任务专用前后处理、一次几何变换和输出路由。它不是原 Turbo 9 步聚合节点的更名版本，也不是已部署的 GPU 服务。

## 文件与版本边界

- [UI 工作流](../examples/qwen21-pruna-v4.json)：可展开分组，模型、采样器、标记、编码、恢复均可检查。
- [API 工作流](../examples/qwen21-pruna-v4-api.json)：使用明确输入名，不依赖控件序号。
- [v3 原始基线](../examples/qwen21-pruna-v4-baseline.json)：原文件副本，便于复现；原 Downloads 文件不覆盖。
- [构建器](../scripts/build_qwen21_pruna.py)：读取指定 v3 并另存新文件；输出已存在时拒绝覆盖。
- [工作流合同测试](../tests/test_qwen21_pruna_workflow.py)：检查模型参数、单一路由、API 连线、源文件保护和真实 RGBA PNG 保存。

节点插件需要包含本次新增的 `SwwanQwen21Pruna*` 节点。仅 `git pull` 而不重启 ComfyUI、刷新浏览器，已有页面不会自动获得新接口。

## 运行依赖

| 分支 | 必需接口 / 依赖 |
| --- | --- |
| 公共加载和采样 | 支持对应Qwen2.1权重的ComfyUI，以及`ModelAttentionBackend`、`ManualSigmas`等核心节点；权重文件由使用者准备 |
| official编码 | 注册`TextEncodeQwenImage21`且支持`resolution=0`和动态`images`的ComfyUI版本；缺失时报错，不偷偷换编码器 |
| cutout / BFS的auto编码 | QwenImageEditUtils的`QwenImage21ModelConfig_EditUtils`和`EditTextEncode_EditUtils`；原生RGBA能力仍取决于所用Qwen2.1 VAE |
| auto局部校色 / 显式goohai | Goohaitools `GH_CropRestore`，带颜色还原强度接口 |
| 显式bluebox且strength>0 | ComfyUI-BlueBox `ColorCalibratorBlueBox` |
| BFS专用链 | 冻结BFS工作流所用检测、参考对齐、RMBG等插件及模型，详见既有[统一工作流依赖](QWEN21_UNIFIED.md)；通用任务不执行此链 |

本机开发仓库与实际ComfyUI安装目录可能是两个副本。应把本次插件版本安装到真正启动的`custom_nodes/ComfyUI_Swwan`后重启；本次只使用隔离目录验证，没有自动覆盖已有运行副本，也没有下载模型。

## 首次运行

1. 在 Models 区确认三个基础模型及 Pruna LoRA 路径存在。此处不自动下载模型。
2. 在 `01 / 任务、预算、恢复与附加 LoRA` 选择任务；默认 `edit`，保留 v3 的额外编辑提示词、种子和 edit-only 附加 LoRA 配置。
3. 在 `02 / 文件名或外接输入` 填写已上传到 ComfyUI input 目录的主图文件名。示例中的文件名来自 v3，必须换成当前实例已有文件。
4. 局部任务单独填写白色选区遮罩文件名；需要保护时另填保护遮罩。所有遮罩尺寸须与主图一致。
5. 双图编辑或 BFS 在 reference filenames 每行填一个参考图文件名，顺序有意义。
6. 运行并查看最终保存结果；诊断中核对原生裁剪尺寸、工作画布及有效预算。先用固定种子做前后对照。

`Inputs` 也提供可选 IMAGE/MASK/参考图端口，方便接外部加载器：外接输入优先，不会再把文件名内容合并进来；不用的文件名建议清空。文件遮罩读取 **red 通道**，不是 `1-alpha`；软灰度可作为部分选区，白色为完全选中。纯文生图分支不读取主图、遮罩或参考图，未选任务也不应加载检测器或执行采样。

**LoadImage 的 MASK 为 `1-alpha`，不是自动物体选区。** 不要把透明背景误当成要移除的主体。画圈/涂抹选区、保护区和生成 alpha 是三个不同数据。

## 任务输入合同

| task | 必需输入 | 遮罩含义 | 参考图 | 输出 |
| --- | --- | --- | --- | --- |
| `edit` + `full` | 主图、编辑提示词 | 忽略选区 mask，整图可编辑；保护区另行提供 | 可选，有序 | 恢复尺寸的编辑图 |
| `edit` + `local` | 主图、编辑提示词、非空 mask | 白色为允许编辑区域，黑色保留 | 可选，有序 | 原图尺寸，范围外原图 |
| `outpaint` | 主图、至少一侧扩展量 | 自动产生新增画布区域；不是上传物体 mask | 不使用 | 扩展后的尺寸 |
| `cutout` | 主图、非空目标 mask | 定位要保留的主体，不是删除区，也不是最终 alpha | 不使用 | 原尺寸透明 RGBA PNG |
| `watermark` + `local` | 主图、非空水印 mask | 指定移除区域 | 不使用 | 原图尺寸，范围外原图 |
| `watermark` + `full` | 主图 | 整图允许模型查找水印 | 不使用 | 原图尺寸；不保证未画区域零变化 |
| `face_swap` | 主图、参考人物图；可选手动目标 mask | 目标脸/头区域，或由原 BFS 检测链生成 | 1..6 张，与固定人物顺序匹配 | 逐人恢复后的原图尺寸 |
| `photoreal` + `image` | 主图、提示词 | 整图生成/写实重绘 | 不使用 | 图生图结果 |
| `photoreal` + `text` | 提示词、宽高 | 不使用任何图片/遮罩 | 不使用 | 指定尺寸的纯文生图 |
| `remove` | 主图、非空 mask | 白色为待移除对象；黑色保留 | 不使用 | 原图尺寸，范围外原图 |
| `angles` | 主图、角度参数 | 不是局部贴回任务 | 不使用 | 视角变化图，不强行回贴旧内容 |

局部模式选区为空、全部被保护或参考图不足时应在采样前报错，不静默改成整图编辑。纯文生图只有 `photoreal + generation_mode=text` 生效，不能仅切换空 latent 而仍把参考图送入编码器。

一次只接受一张主图，不接受图像 batch；参考图列表最多6张。输入 RGBA 的 alpha 会保留在需要还原原图的任务中，但不会被自动转换为选区。抠图任务不接受非空保护 mask，保护原图内容与生成透明背景不是同一个合同。

## LoRA、提示词与原 v3 的关系

公共模型链为：

`UNETLoader → Pruna LoRA（原权重1）→ 可选附加 LoRA → 当前任务 LoRA → 原 ModelAttentionBackend`

| 任务 | v3 任务 LoRA | 默认权重 | 固定指令/控制 |
| --- | --- | ---: | --- |
| edit | 无任务 LoRA | 0 | 用户编辑提示词 |
| outpaint | `Qwen2.1_扩图_GH_3750.safetensors` | 1.0 | 自然延展灰色区域 |
| cutout | `心宝强化抠图.safetensors` | 1.0 | Xinbao 主体提取、遮挡补全、透明 PNG |
| watermark | `Qwen21-水印移除watermark_remover_v2.safetensors` | 0.8 | 移除水印 |
| face_swap | `bfs_head_v1.1_qwen_2.1.safetensors` | 0.7 | 主图/参考图角色明确的 head_swap 指令 |
| photoreal | `pipi-qwen2.1-PhotoReal10.safetensors` | 0.8 | photorealistic + 用户要求 |
| remove | `Qwen2.1_万物移除Remove_GH.safetensors` | 1.0 | 移除绿色标记区域 |
| angles | `angles_v2_full_qwen21_multiple_angles_v2_qwen21_multiple_angles_v2_000001500.safetensors` | 0.9 | 方位、俯仰和特写指令，不再沿用绿色移除指令 |

路径默认保留 v3 中的 `qwen21/` 前缀，Controls 中的 `task_lora_*` 可修改为当前模型目录下的实际路径。`lora_strength=-1` 表示使用对应任务默认权重，非负值为显式覆盖；任务路径为 None/空时按节点合同处理，不能宣称加载成功。

v3 的节点530附加 LoRA不再无条件作用于所有工具任务。默认 `additional_lora_mode=edit_only`，保留其原文件名及0.6权重用于编辑；其他任务不叠加。需要纯净编辑时关闭附加 LoRA。原始基线文件保留 v3 的全局叠加行为，不能把新默认与旧基线视为完全相同配方。

额外提示词会拼到任务指令后；修改任务时要检查上一次的额外要求是否仍适用。提示词增强不自动开启，不增加额外 LLM 请求。

`negative_prompt` 默认空；官方编码路径接受该字段，但原 v3 的 `CFG=1` 保留，因此不能把设置负面提示词理解为保证产生负向引导效果。EditUtils 路径沿用 `ConditioningZeroOut`，非空负面提示词会明确报错，而不是静默忽略。

### 共享不等于 GPU 常驻适配器

任务路由只激活当前 LoRA，不通过“把全部 LoRA 串起来再设0”假装实现 GPU 常驻。当前节点仍使用 ComfyUI 原生模型 patch/缓存管理；模型对象缓存命中不等于独立适配器驻留显存，也不等于切换零开销。INT8 + GPU 常驻切换需要单独实现验证及完整热态 `A→B→A` 计时，不属于本地工作流加载验收结果。

## 三条共享处理规则

### 面积预算

`qwen_min_resolution=128`、`qwen_max_resolution=1024` 是默认面积预算边界，非最长边或固定正方形尺寸。

`原图有效mask → 外接框 + 上下文 → 原生W×H → sqrt(W×H)最近32档 → min/max夹取 → BlueBox候选 → 同步缩放/补边`

- 下限向上、上限向下对齐32；对齐后下限大于上限明确报错。
- 先裁局部再降低推理尺寸，原图保留用于最终恢复；不先把整张原图缩到1MP。
- 工作画布宽高为32倍数，某一维可以超过预算数字。
- `context_factor=1.3` 表示每侧增加选区外接框短边的0.3倍上下文，不是把整张图乘1.3；最终编辑权限仍由原选区控制。BFS沿用其专用检测/裁剪上下文，不使用这个通用因子。
- 低于512预算的缩小遮罩保留面积覆盖灰度，禁止下游无条件以0.5阈值再次丢弃小选区。
- 128是可用下限，不是生成质量保证。发丝、面部、文字、小物体可能需要更高预算。
- 多边扩图的新增区域外接框常常接近整张新画布，不能承诺只推理几条窄边。

### 位移与几何

新工作流移除 v3 的 `Resize → Cut → 编码器固定992` 三次独立定尺寸；共享前处理只记录一次几何。官方编码路径用已对齐图和 `resolution=0`；latent来自同一编码器。EditUtils路径保留独立编码合同，但要求输入已经对齐，返回主图尺寸不变、额外padding为0。`Unpad`目前是画布/记录一致性守卫，不会猜测并裁切未知补边；发现二次缩放、额外padding或解码尺寸不一致会报错，之后才由恢复器执行唯一几何逆变换。

参考图独立按 `reference_resolution` 的面积预算缩小，并在右侧/底部补到32倍数，不放大较小参考、不套用主图BOX。EditUtils抠图路径请求主图仅用于VL；部分EditUtils版本会自动把它提升为reference，编码记录中的 `cutout_reference_promoted` 会标明实际情况，不宣称能绕过该插件行为。

输出尺寸一致、选区外像素保护、模型选区内结构保真是不同验收目标。BlueBox与SEAM解决的是管线尺寸与坐标，不承诺模型生成内容零位移。不同人物参考图不能套用主图裁剪框，只有同场景同坐标的图片才可共用几何。

### 色差与保护

- `color_mode=auto` 仅对 `edit/local`、`watermark/local`、`remove`、`outpaint` 选择GH；通用整图任务默认不校色，不把GH、BlueBox、肤色匹配全开叠加。
- GH路径保留原生颜色还原，干净原图为参考，绿色/红色标记不得成为颜色基准。
- BlueBox可选用于整图/编辑校色；自动差异mask不是用户授权编辑mask。
- BFS保留其独立的 `skin_match` 肤色匹配控制及人物融合链，不走通用恢复器的 `color_mode`。
- 改色、改光照、风格化、多角度通常应关闭强制原色还原。
- 最终保护使用原图坐标有效mask；恢复器的羽化mask、输出alpha和保护mask不可互换。

原生 GH/BlueBox 是对应分支的外部运行依赖；未选择时不应导入其模型或运行其处理。缺少选定校色插件应明确报错，不静默换成另一个颜色算法。

### 常用参数速查

| 参数 | 默认 | 作用和边界 |
| --- | ---: | --- |
| `qwen_min_resolution` / `qwen_max_resolution` | 128 / 1024 | 主图或局部裁剪面积预算下限/上限，32档；不是最终输出尺寸 |
| `qwen_tolerance` | 1.0 | BlueBox宽高候选筛选的面积容差，不是模型位移精度 |
| `reference_resolution` | 768 | 单张身份/编辑参考图的独立面积预算，32倍数 |
| `color_strength` | 1.0 | 仅通用GH/BlueBox校色强度，不改变alpha或BFS肤色匹配 |
| `expand_percent` / `feather_percent` | 4 / 5 | GH及BFS恢复扩展/羽化；无校色、BlueBox通用恢复使用羽化但不扩大权限 |
| `marking` | brush | 通用局部选区为涂抹；circle必须闭合并填充内部；BFS不使用此开关 |
| `enhanced` / `block_size` | false / 32 | 仅移除：原图选区块化，随后重新扣除保护；block_size不是Qwen对齐倍数 |
| `mark_opacity` | 0.4 | 普通移除绿色标记透明度；增强移除固定1，抠图红色内轮廓固定1 |
| `padding_unit` / 四向边距 | pixels / 128,128,0,0 | 仅扩图，顺序left,right,top,bottom；percent按原图对应轴计算 |
| `seam_band` | 0 | 仅扩图，允许修改旧画布内侧接缝带的像素宽度；0严格保留旧区域 |
| `people` / `swap_head` / `forehead` / `skin_match` | 1 / false / false / false | 仅BFS人数、换头范围、额头范围、肤色匹配；LoRA固定v3的v1.1默认路径 |
| `skin_strength` | 0.25 | 仅BFS，开启skin_match后控制原生肤色匹配混合强度，不等于通用color_strength |
| `output_width` / `output_height` | 1024 / 1024 | 仅photoreal纯文生图，32倍数，不受局部裁剪面积预算裁小 |
| `azimuth` / `elevation` / `close_up` | 0 / 0 / false | 仅angles视角提示词：水平角度30度档，俯仰0/30/60/90度 |

隐藏或当前任务不用的参数不等于删除：切换回来仍保留原值。切换任务时至少复查 `extra_prompt`、保护区、参考图和附加LoRA模式。

## 专用任务说明

### 移除与局部编辑

`marking=brush` 使用涂抹选区并保留软灰度；画圈模式填充闭合区域，不闭合则报错。`enhanced` 明确控制块化，原图坐标保护应在扩展后再次扣除。普通绿色覆盖保留默认不透明度0.4，不因整合提升为1；主动开启增强移除时才固定为1。

输出原图尺寸，局部范围外从实际加载原图取回。范围外像素不变不代表移除区域内部已生成正确，需检查原始生成块及最终图。

### 扩图

`left/right/top/bottom` 与 `padding_unit` 决定四向扩展；新增区域用灰色标记，原图保留。`seam_band` 明确允许旧图边缘参与接缝融合的宽度，0表示原图区域严格保护。扩图校色要检查旧图接缝的一致性，不能把灰色占位当成希望生成的颜色。

RGBA扩图保留旧画布alpha，新增区域alpha为1；校色仅处理RGB，之后恢复这份alpha。`color_strength=0`会跳过BlueBox校色，而不是仍运行其附加颜色处理。

目前复用原生GH统计与贴回链，没有保证其颜色统计完全排除新增画布的灰色占位；不要将最终原图保护误写成“校色统计只看旧图”。扩图的原始块、接缝及大面积新区域仍需GPU实图验收。

不能为了“减少推理像素”默认拆成四次独立生成，否则会改变全局连续性与调用次数。

### 抠图与遮挡补全

主体mask用于定位和裁剪，红色轮廓用于提示，不是最终alpha。`auto`编码采用EditUtils兼容路径；输出必须实际有4通道，RGB结果应报错。恢复时将完整生成的RGBA放回透明原尺寸画布，不用原选区裁掉补全部分。

保存使用 `SwwanSaveImage` 的 `png / straight / alpha_mode=auto`。PNG只是容器格式，只有真实alpha才是透明输出；验证要重读PNG检查RGBA、透明背景和半透明边缘。

### BFS 换脸／换头

参考图逐行对应检测后固定的人物顺序：从上到下、同行从左到右；至少提供所需人数的参考，缺少时报错。`people`为1..6，人物顺序来自原图，不在每次生成后重新检测和重新排序。当前BFS使用v3的v1.1路径，不提供会改变整套链路合同的版本切换。

保留参考脸对齐、RMBG、脸/头范围、肤色匹配和逐人恢复。每人调用共享Pruna核心一次，多人不是一次采样完成。非BFS任务由lazy结果路由跳过该分支；不要为了看预览而无意启用未选分支的输出节点。

### 写实与多角度

写实图生图保留主图条件；纯文生图仅文本与输出宽高，不读取Inputs里的旧图片文件名。尺寸需要满足当前Qwen编码器/VAE合同。

多角度改变结构，不使用局部范围外硬贴回来覆盖新视角。角度参数生成实际任务提示词，可在Actual task prompt查看。跑完不等于角度、材质、身份全部正确，必须视觉配对验收。

## 工作流节点与调试

| 节点 | 用途 | 检查重点 |
| --- | --- | --- |
| 600 Controls | 单一任务、种子、预算与参数入口 | task、实际prompt、LoRA及附加LoRA |
| 601 Inputs | 按需读取主图、选择、保护、参考 | 文件名、同尺寸、mask/alpha语义 |
| 602 Model | 对干净Pruna模型应用当前任务adapter | 不累积上一任务LoRA |
| 603 Prepare | 选区、预算、上下文、SEAM | 原生/工作尺寸、是否空选区 |
| 604 Mark | 在副本绘制任务标记 | 干净参考不能被覆盖 |
| 605 Encode | official/EditUtils兼容编码 | 图序、latent尺寸、padding |
| 522/523/524/525/526 | 保留的Pruna采样链 | 固定六步sigmas、Euler、CFG1、同一seed |
| 606 Unpad | 验证编码画布与零额外padding | 不猜测裁边，不重复去共享几何padding |
| 607 Restore | 校色、逆变换与最终保护/RGBA | 范围外零变化、真实alpha |
| 608 Face | BFS专用逐人动态子图 | 固定人物序、参考映射、逐人结果 |
| 609 Output | lazy选择最终分支 | 不同时执行通用与BFS采样 |
| 610 Save | 保存最终RGB/RGBA | 重读实际PNG，不只看UI预览 |

标记图预览默认禁用，避免BFS模式下强制执行通用前处理；仅在调试对应分支时启用。Actual task prompt预览是轻量控制输出，不额外执行图片推理。

## API使用

文件名输入的选区和保护图用 Pillow 按 EXIF 方向读取红通道，精确保留 `0/255=0`、`255/255=1` 及软灰度；不走视频解码器的 RGB 范围变换。外接 MASK 端口仍原样使用，调用方必须保证全白保护为1；接近1的灰色不是绝对保护。主图与参考图保持 ComfyUI 原有解码，不改动其像素合同。

API JSON保留具体文件名、LoRA路径和参数，需要按实例实际文件替换。任务参数位于节点600，文件输入位于601，模型路径仍在原节点509/527/548/511。更新600的seed同时驱动通用RandomNoise与BFS，不再维护两个随机种子。

API客户端可以修改任务、mask文件、参考列表等，但不要通过替换节点序号来猜端口含义；UI与API以命名输入和真实连线互相核对。任务切换后必须检查上一任务的额外prompt、保护区与参考是否仍然适用。

## 验证边界

本地静态、CPU几何、图展开、保存重载或模拟编码测试，不代表GPU生成质量通过。旧Turbo工作流的效果也不自动继承到Pruna六步。

2026-11-05已完成L20真实推理：8类任务、20个场景；发现并修复文件遮罩白色经视频解码变为接近254/255的问题后，复测12例。全白保护16,200像素零变化、扩图旧画布零变化、RGBA原alpha保持、抠图真实透明、A→B→A重新采样后A逐像素一致。详见 [L20验收与限制](QWEN21_PRUNA_L20.md)。

执行和像素合同通过不代表模型质量全部通过。多角度、去水印细节、无上下文极小裁剪和整图文字限定范围存在未通过样例；未完成同块校色对照、内部位移测量、6人或GPU常驻adapter验收。仍不承诺128预算质量、语义零位移、多参考身份一致性或切换零开销。

代码已本地提交，GPU使用隔离实例；没有Git推送、升级用户原运行副本或部署线上服务。
