# 展开式 Turbo 移除：BlueBox 预处理 + 原生孤海贴回

[可编辑工作流](../examples/qwen21-remove-turbo-bluebox-gh-expanded.json) 使用 Turbo INT8 + Remove GH 1.0 + Sage、9 步完整 sigma；各处理阶段均为独立节点，不使用 Dispatch/Core。

双模式扩展见 [普通 / 增强说明](QWEN21_REMOVE_ENHANCED.md)；本页普通版示例保留作为回归基线。

## 本次新增能力

`SwwanCropByMaskV5` 在 `crop_mode=qwen21_aligned` 下增加 `qwen_restore_backend`：

- `swwan`：保持旧缺省行为，SEAM 接 SwwanRestoreCropBoxV4。
- `goohai`：以孤海 SEAM 合同表达同一套前处理，接原生 **GH_CropRestore**。完整保留其颜色还原、羽化、扩展功能。

原有前9个输出的顺序与恢复后端默认值未变；动态预算新默认见下文。`ops/qwen21_gh_seam.py` 只转换元数据，不进行像素恢复，不移植颜色算法，不调用 BlueBox/Swwan 贴回。GH 先去除目标补边，再恢复到原裁切区域，校色后融合。

专用工作流选择 `qwen_resolution_mode=crop_area`、tolerance=1.0；历史示例显式保存的 `qwen_max_resolution=1920` 不随节点新默认改变。当前节点预算默认 **min=128 / max=1024**，仅动态模式生效；旧文件缺失的字段采用新默认，需要复现旧512下限时请显式设置512。保留候选尺寸评分和严格不动点校验，宽高为32倍数。图像和遮罩一次同步变换，完整保留选区。编码器接 `encoder_resolution=0`，采样直接使用官方编码器latent，后面不再加Resize。

注意 tolerance=1.0 在 resolution=1024 的候选面积容差为 3.125%，不是偏移精度。多张图分别通过候选算法不代表它们的尺寸或空间坐标一致。

## 动态面积预算

`qwen21_aligned` 已包含 BlueBox 的 32 倍数候选、输入比例/常见比例/面积评分和编码不动点校验，不需要再串接 `Cut (Qwen-Image 2.1)`。它不是原 Cut 的 crop/pad/stretch 三模式复制：先扩展真实上下文、必要时补边，图像和 mask 共用几何记录，保留完整选区。实现还会先筛除非不动点再评分；原 Cut 是先评分、未通过时再按比例误差回退，所以不承诺所有输入都选到完全相同的宽高。

先在原图坐标中取完整 mask 外接框并加上下文（限制在原图可用范围），得到 W×H；在任何缩放、BlueBox 比例扩展或补边前计算：

```text
R = 32 * floor(sqrt(W * H) / 32 + 0.5)
minimum = 32 * ceil(qwen_min_resolution / 32)
maximum = 32 * floor(qwen_max_resolution / 32)
resolution = clamp(R, minimum, maximum)
```

`qwen_min_resolution` 追加在旧控件末尾，默认 **128**，可设32–2048；`qwen_max_resolution` 默认 **1024**，保留原512–2048输入范围。新节点和缺失字段使用新默认，显式保存/连线值优先。下限向上、上限向下取32倍数：例如257→288、1560→1536；对齐后下限大于上限会明确报错，不静默交换或突破上限。128等低于512的预算仍属实验，仅有CPU几何验证，不代表模型质量或速度已经验收。

低于512的实际预算在缩小遮罩时使用面积覆盖采样，避免最近邻/点采样遗漏细线和离散小选区。原来二值的边缘可能变成灰度覆盖，不强制二值化或提升遮罩强度；若缩放后仍完全丢失选区，节点报错而不继续推理。512及以上预算保留既有遮罩采样，图像缩放、几何、恢复与校色算法不变。非零覆盖不代表低预算下模型能识别细节，最终硬保护仍应使用原图坐标的有效mask。

它们是面积预算边界，不是最长边或最短边；例如 min=256、max=1560 时，小裁剪可用约 256² 预算，大裁剪最多取 1536 档，长条画布的单边仍可能大于该数值。预算不使用白色像素总数或整张原图面积。动态值交给已有候选及不动点校验；编码仍为 resolution=0，GH 贴回与颜色还原不变。

节点默认模式保持 `fixed` 以兼容旧工作流；专用工作流显式选择 `crop_area`。动态模式显示 min/max、隐藏固定 resolution；固定模式隐藏 min/max 且不应用二者，其他裁剪模式也不受影响。输出不变：9 原生裁剪宽、10 原生裁剪高、11 实际预算；前 9 个输出不变。预览节点 37/38/39 显示这三项，30/31 显示最终工作画布。更新节点并重启 ComfyUI、刷新前端后，在 aligned + crop_area 模式查看新控件；下载的旧 JSON 无需改位序或重连输出。

## 界面参数速查

节点帮助已覆盖三种模式、动态公式、输入遮罩语义、首帧/空选区、恢复接线和兼容边界；全部输入与12个输出均有中文悬停提示，不只在本文件说明。

| 参数/输出 | 作用与生效条件 |
| --- | --- |
| crop_mode | bounds传统裁剪；edit_region局部编辑；qwen21_aligned启用BlueBox与面积预算；后两者仅首张 |
| edit_expansion_mode / edit_*_factor | 编辑模式；每侧增加选区框短边S×(factor−1)，不是整体宽高乘倍数 |
| *_reserve_ratio | bounds按对应宽高、受reserve_max限制；编辑模式reserve_ratio按短边S计算，不受该上限限制 |
| fill_mask_holes | 填内部孔洞，可能扩大选区；不等于保护mask |
| qwen_resolution_mode | fixed用固定qwen_resolution；crop_area按局部面积动态选择128–1024默认区间 |
| qwen_tolerance | 候选面积容差，不是位移误差；不动点校验仍执行 |
| qwen_restore_backend | swwan接SwwanRestoreCropBoxV4；goohai接GH_CropRestore；不在裁剪节点执行校色 |
| region_info / region_mask | 同源恢复几何记录/同步遮罩；不能混用SEAM后端或只靠BOX猜恢复坐标 |
| encoder_resolution | aligned输出0接官方编码器resolution；采样使用该编码器latent |
| native_crop_width/height | 比例调整/缩放/补边前的局部宽高，是动态预算来源 |
| width/height / working_resolution | 实际模型画布宽高/面积预算R，两者不是同一数值 |

bounds的detect/reserve_mode/round_to_multiple/batch_mode/device不用于aligned；edit_region的output_size/custom_width/custom_height/alignment也不用于aligned。减少管线几何误差不等于模型内容零位移；严格保护和颜色校正仍由后续恢复/合成处理。

## 工作流使用

1. 节点 1 加载干净 RGB 原图；2 加载同尺寸实心移除 mask（红色通道，白选黑保留）。普通 LoadImage MASK 是 1-alpha，不是自动选区。
2. 节点 4 默认全黑保护；需要时换成原图尺寸的保护 MASK，白色保护。节点 5 先从移除区域扣除保护。
3. 节点 6 的 `qwen_restore_backend` 必须为 `goohai`。节点 7 是对齐后的 40% 纯绿色覆盖。节点 8 追加用户提示词，32 预览最终文本；PE 不执行。
4. 节点 21 为原生孤海恢复：颜色还原 **1.0**，0 关闭；扩展 4%、羽化 5%。节点 10 的 LoRA 强度是另一项控制。
5. 节点 34 最后按原选区有效范围融合，确保选区外及全白保护区使用原图像素。GH 内部保留软边，最终范围使用二值支持，避免软 mask 重复相乘。
6. 节点 22 / 23 / 25 / 26 分别保存实际模型输入、原始解码块、最终门控 mask、最终图。门控 mask 不是 GH 内部的羽化权重。

现流程为 RGB，不承诺保留 alpha。空选区报错。SEAM 格式不可混接；更新节点后需重启所使用的 ComfyUI 实例。

## 依赖

- ComfyUI 官方 Qwen Image 2.1 编码、loader、采样和保存节点。
- ComfyUI_Swwan（本次兼容更新）。
- Goohaitools-comfyui 原生 GH_CropRestore，需包含颜色还原强度参数。
- ComfyUI-KJNodes 的 ManualSigmas；Comfyui-Qwen-Image-2.1-BlockCache-T8 的 QwenImage21SageAttentionT8（不启用 BlockCache）。
- Turbo INT8 DiT、Qwen 3VL INT8 CLIP、Qwen 2.1 VAE、Qwen2.1_万物移除Remove_GH.safetensors。

## 验证边界

CPU 用冻结的原生孤海恢复类核验奇数尺寸、贴边、小选区、多块选区、软 mask、坐标恢复、颜色开关及 PNG 重读。前端保存重开后的 API 与构建器一致。

动态版在 L20 隔离实例重新生成 2 素材 × 3 seed，并另做一次重叠保护测试。原生裁剪分别为 133×113、443×355，两者都取最低预算 512；工作画布为 576×480、576×448，中位耗时 3.035 / 2.073 秒。小物体及选中水印移除通过这两例检查，所有选区外像素及 1761 个保护像素不变。颜色、标志、材质细节仍可能被重建。

固定 1024 的历史数据仅作对照。大于 512 的 GPU 档位尚无这次效果验收；800、1920、半档上取整和非 32 倍数上限等通过 CPU 规则测试。
完整说明、API 脚本和图片报告在配套 `replicate-models/image/qwen21-eraser-native/BLUEBOX_REMOVAL.md` 与 `validation/dynamic-budget-20261029/`。未替换线上服务。
