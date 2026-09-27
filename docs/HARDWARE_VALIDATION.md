# 硬件独立验收

本轮本机为 macOS CPU，无 CUDA；没有下载模型、执行 Qwen 推理或 GPU 补丁。CPU 结果仅覆盖图片算法和注册合同。下面项目保持待验，不能从 CPU 通过推断。

| 功能 | 独立检查与证据 |
| --- | --- |
| 普通 Resize GPU | 记录 GPU/torch/CUDA/输入尺寸/dtype，核对输出尺寸和遮罩；对比 CPU，记录允许的插值误差 |
| RTX VSR | 支持的 NVIDIA 环境和 nvvfx 版本，8 像素对齐、DLPack 生命周期、批次/遮罩、异常 cleanup、显存，真实输出文件 |
| SageAttention | 匹配 GPU 架构 wheels、版本、补丁生效和 fallback；完整推理结果及显存/速度，不只导入 |
| MiniMax H3 | 模型/Comfy版本、sm 架构、SM90 V 对齐、缓存与形状、输出等价和峰值显存；不在基础安装强制加载 |
| Qwen2511 | 固定模型/LoRA、输入/遮罩、prompt、seed、sampler、steps、CFG、dtype，成对生成；检查裁剪还原边界和实际图片效果 |
| Webcam / Screen | 授权设备环境、有效帧/停止行为、平台依赖；注册成功不等于设备捕获成功 |

百万像素节点的委托调用与零值旁路已在 CPU 合同测试中验证；GPU 执行仍须上述证据。第三方插件共装仅验证注册归属，本轮没有验证其可选 Triton/guidedFilter 节点。
