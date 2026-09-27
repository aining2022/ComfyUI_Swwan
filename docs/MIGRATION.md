# 迁移与数据合同

通用迁移器支持 UI workflow（nodes/links）以及 API prompt（class_type/inputs），也支持带 `prompt` 字段的封装。默认检查后另存 `<原名>-swwan.json`，禁止覆盖输入或既有输出。`--dry-run` 只报告。

未知来源的重名节点会列为 ambiguous：必须通过 `--swwan-node-id 编号` 指定迁移范围。已有 `cnr_id=comfyui_swwan` 或 `swwan_version` 可确认本仓来源。API prompt 使用 `_meta.swwan_version`。只在文件全部为本仓历史节点时显式 `--assume-swwan`。未选择的第三方节点不修改。

重命名不改节点编号、连线、布局或旧参数；新可选控件和输出追加，不重排旧端口。需要补齐新增控件时传 `--comfyui-root`，使用实际接口。旧字体值转换为 FreeMono；Resize v2 的历史非法 keep_proportion=false 修复为 stretch（不改有效枚举值）。二次迁移结果相同。完整旧 ID 和替代关系由 [节点目录](NODE_CATALOG.md) 生成。

Qwen2511 专用迁移器会按源接口重映射裁剪 SEAM/MASK 和相关端口，补接原图／BOX；再用通用工具补齐现有新模式控件。示例是迁移结果，核心、QwenEditUtils、模型、提示词和采样不变。

专用迁移器同时保存当前输入名对应的 `widgets_values_named`，覆盖原插件的旧字段名；Swwan 模式控件在加载时按名字恢复。不同前端可能将 COLORCODE 显示为控件或仅保留插槽，不能仅凭 `widgets_values` 的长度确认兼容。验证时还需用实际前端转成 API prompt，检查转换后的枚举、数值及颜色连线，并测试保存重载。更新插件后刷新浏览器以加载新脚本。

## 不可互换的数据

| 类型 | 语义 |
| --- | --- |
| BOX | 共享裁剪矩形；本仓旧 Crop/Restore 协议 |
| BBOX | WAS 逐图边界列表 |
| IMAGE_BOUNDS | KJ 行列边界，包含其端点约定 |
| SEAM | 编辑区域原图/原遮罩/越界填充/缩放/接缝完整信息 |
| STITCH3 | Resize Sum/恢复尺寸流程的原协议 |
| IMAGE tensor batch | 同一尺寸的 NHWC tensor |
| IMAGE list | 可包含不同尺寸；不能直接当同尺寸 batch |

Legacy 保留原算法、原输出顺序和批次规则。新主入口不是每个历史合同的替换：有 replacement 的条目表示推荐选用入口，自动迁移不将 Legacy 类型强行替换为主入口。


## 已编辑换脸／换头文件的修复精简

本次 179 节点／197 连线版本作为独立基准保存在 `tests/fixtures/qwen-face-head-current.json`。
交付的界面副本是 [`examples/qwen-face-head-final-swwan.json`](../examples/qwen-face-head-final-swwan.json)，为 145 节点／167 连线；不附带 API 文件。

```sh
python scripts/repair_qwen_face_head_workflow.py edited-swwan.json new-fixed.json --comfyui-root /path/to/ComfyUI
```

工具只适用于该已编辑工作流，拒绝覆盖源文件或已有目标。它从 SaveImage #163 反向解析显式连线及同名 Set/Get，保留选择器的全部输入分支；不根据当前布尔值、索引或模型配置裁掉可切换分支。循环结束 #162 的展示状态 initial_value2/3 断开，端口位置保留；关闭缓存且旁路的 #165 直接重连图像，进度链、预览、说明面板及不可达配置删除。输出命名仍为 `换脸_无高清`。

修复依据：#92 保留 768×768、Essentials、keep proportion、always、lanczos、整除数 0；#62 的旧“颜色、设备”接口补默认 opacity=1.0，同时支持“颜色、透明度、设备”。#47/#58 的丢失背景色依据原工作流恢复为 #ffffff，其他合法值保留。仅识别确定的错位特征，不对未知坏值进行猜测。非法枚举、数值／布尔类型、颜色、必填参数或端口、连线及虚拟变量会明确报错。

`web/js/swwan_widget_values.js` 为全部 Swwan 分类节点按实际输入名保存与恢复参数，隐藏和转换为输入的控件均不依赖位置匹配；保留 ComfyUI 原生位置数据和界面附加控件。COLORCODE 控件缺席时，合法颜色仍保存并注入执行参数；实际连线优先。历史文件已有错误命名映射时须先运行修复工具，新前端不会把错误值静默改成默认值。安装这些前端修改后须刷新页面，再导入修复副本。

验证：5 项定向修复测试、7 项既有图片参考测试、7 项 Qwen 图片合同测试；独立 EasyUse 真实图展开执行三次模拟裁剪／编辑／还原，并经 SaveImage 写出后重读检查三处累计结果。真实前端在 COLORCODE 控件存在／缺席两种情况下验证转换为输入、颜色连线、命名保存重载及 CPU 实际输出，证据见 `tests/fixtures/face-repair-browser-acceptance.json`。未下载模型或执行 Qwen／检测模型／CUDA。


### 反向代理路径下的前端修复

部署在 `/comfyui/` 等子路径时，旧扩展使用 `/scripts/app.js`／`/scripts/api.js` 会请求网站根目录，产生 404；因此颜色控件及按名称恢复逻辑根本未加载，后端仍收到错位值。全部 Swwan 前端现使用相对于 `extensions/ComfyUI_Swwan/` 的 `../../scripts/` 导入，保留部署路径。此修复不改变节点算法或接口。

更新插件后重启服务并强制刷新页面，重新导入修复精简版。已经在失效前端中另存过的错误参数不能作为新的正确基准。`tests/test_frontend_base_path.mjs` 检查根路径和 `/comfyui/` 下的真实导入地址；前端验收直接加载迁移文件的完整控件列表，而非只测试浏览器自身保存的列表。
