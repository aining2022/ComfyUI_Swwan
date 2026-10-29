# 展开式 Turbo 移除：BlueBox 预处理 + 原生孤海贴回

[可编辑工作流](../examples/qwen21-remove-turbo-bluebox-gh-expanded.json) 使用 Turbo INT8 + Remove GH 1.0 + Sage、9 步完整 sigma；各处理阶段均为独立节点，不使用 Dispatch/Core。

双模式扩展见 [普通 / 增强说明](QWEN21_REMOVE_ENHANCED.md)；本页普通版示例保留作为回归基线。

## 本次新增能力

`SwwanCropByMaskV5` 在 `crop_mode=qwen21_aligned` 下增加 `qwen_restore_backend`：

- `swwan`：保持旧缺省行为，SEAM 接 SwwanRestoreCropBoxV4。
- `goohai`：以孤海 SEAM 合同表达同一套前处理，接原生 **GH_CropRestore**。完整保留其颜色还原、羽化、扩展功能。

原有前 9 个输出的顺序和既有默认值均未变。`ops/qwen21_gh_seam.py` 只转换元数据，不进行像素恢复，不移植颜色算法，不调用 BlueBox/Swwan 贴回。GH 先去除目标补边，再恢复到原裁切区域，校色后融合。

专用工作流默认 `qwen_resolution_mode=crop_area`、`qwen_max_resolution=1920`、tolerance=1.0；保留候选尺寸评分和严格不动点校验，宽高为 32 倍数。图像和遮罩一次同步变换，完整保留选区。编码器接 `encoder_resolution=0`，采样直接使用官方编码器 latent，后面不再加 Resize。

注意 tolerance=1.0 在 resolution=1024 的候选面积容差为 3.125%，不是偏移精度。多张图分别通过候选算法不代表它们的尺寸或空间坐标一致。

## 动态面积预算

先在原图坐标中取完整 mask 外接框并加上下文（限制在原图可用范围），得到 W×H；在任何缩放、BlueBox 比例扩展或补边前计算：

```text
R = 32 * floor(sqrt(W * H) / 32 + 0.5)
resolution = clamp(R, 512, 32 * floor(qwen_max_resolution / 32))
```

默认上限 1920，实际最高合法档 1920。它是面积预算，不是最长边。预算不使用白色像素总数或整张原图面积。动态值交给已有候选及不动点校验；编码仍为 resolution=0，GH 贴回与颜色还原不变。

节点默认模式保持 `fixed` 以兼容旧工作流；专用工作流显式选择 `crop_area`。动态模式隐藏固定 resolution 控件。末尾追加 3 个 INT 输出：9 原生裁剪宽、10 原生裁剪高、11 实际预算；前 9 个输出不变。预览节点 37/38/39 显示这三项，30/31 显示最终工作画布。

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
