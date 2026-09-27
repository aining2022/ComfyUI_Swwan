# 安装与验收清单

1. 使用 ComfyUI Python，保留原 torch/CUDA；将本仓放入 custom_nodes，安装 requirements 并重启／刷新。对象信息应为 113 个 Swwan 能力，21 主入口，显示名带 `(Swwan)`。
2. 使用本仓 5 个前端脚本，网络中不应出现 Swwan 对 `/rgthree/` 或缺失相对模块的请求。缺失 vision/color/GPU 可选依赖只在相关执行时报具体错误。
3. 旧重名节点先运行迁移 dry-run，确认来源与编号后另存；检查连线、参数、字体 fallback。原文件保持不变。
4. 按 README 运行 CPU/参考/合同/前端/skill 测试。无模型示例能独立执行。保存文件重读 alpha、格式、元数据、sidecar、绝对路径和编号。
5. 与 KJ/LayerStyle/rgthree 共装，可运行 `scripts/check_coinstall.py --comfyui-root PATH --plugin PATH ...`，分别默认和 `--reverse`；只检查 Swwan 不覆盖第三方的注册归属。
6. 实际浏览器使用隔离服务运行 `tests/frontend_browser.js` 函数，验证模式、颜色连接、动态端口、Seed 元数据和保存重载。独立启动首次 user.css/template 404 不等于插件模块缺失。
7. CUDA/RTX/模型生成和设备功能按 [硬件清单](docs/HARDWARE_VALIDATION.md) 另验。CPU CI 配置已固定 ComfyUI SHA；远程 CI 运行结果须发布后另确认。

本轮证据见 [整改报告](docs/PROJECT_REVIEW.md)。不自动提交、推送或发布。
