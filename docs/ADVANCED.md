# 模型与 GPU 专用功能

以下是已有节点的选用说明。当前 Qwen 图片处理 CPU 验证未验证这些 GPU 路径。

### KJ Alternative SageAttention

`Patch Sage Attention (Swwan)` 用于给 `MODEL` 打上 `SageAttention` attention override；它使用独立节点类型，因此可与 KJNodes 同时安装。

说明：

- 节点本身已内置到 `ComfyUI_Swwan`
- 运行时仍需要额外安装 `sageattention` 或 `sageattn3`
- 选择 `disabled` 时会移除当前模型上的 `optimized_attention_override`

`MiniMax H3 Mem Eff Sage Attention Patch (Swwan)` 会直接替换 MiniMax H3 transformer block 的 attention forward，以降低峰值显存。它需要匹配的 ComfyUI MiniMax H3 API、最新版 `sageattention`、Triton、CUDA 和受支持 NVIDIA GPU 架构。

### NVIDIA RTX Video Super Resolution

`Resize Image (Swwan)` 与 `Image Resize By Megapixels` 都提供 `nvidia_rtx_vsr` 插值方式。该方式按需加载 `nvvfx` / `nvidia-vfx`，仅适用于兼容的 CUDA NVIDIA GPU；输出尺寸会自动对齐到最接近的 8 倍数。


硬件待验项见 [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md)。

[返回项目首页](../README.md)
