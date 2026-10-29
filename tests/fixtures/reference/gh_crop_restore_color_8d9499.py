# SPDX-License-Identifier: GPL-3.0-only
# Frozen Goohaitools-comfyui 8d9499f74f13764182634b02a2b390579491bd21
# nodes/遮罩裁剪V2.py; goohai and contributors. No algorithm changes.
import torch
import numpy as np
import cv2
import math
from PIL import Image

class 孤海_裁剪恢复:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "接缝": ("SEAM",),
                "裁剪图像": ("IMAGE",),
                "遮罩扩展百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
                "遮罩羽化百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
            },
            "optional": {
                "裁剪遮罩": ("MASK",),
                "背景图像": ("IMAGE",),
                "颜色还原强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                                     "tooltip": "与原图对应裁剪区域比较，统一修正返回裁剪图的明度和颜色；0为关闭"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "恢复图像"
    CATEGORY = "孤海工具箱"

    def tensor2pil(self, image):
        """将tensor转换为PIL图像"""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def pil2tensor(self, image):
        """将PIL图像转换为tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def image_to_rgb(self, images):
        """将图像转换为RGB格式"""
        if len(images) > 1:
            tensors = []
            for image in images:
                pil_image = self.tensor2pil(image)
                pil_image = pil_image.convert('RGB')
                tensors.append(self.pil2tensor(pil_image))
            tensors = torch.cat(tensors, dim=0)
            return (tensors, )
        else:
            pil_image = self.tensor2pil(images)
            pil_image = pil_image.convert('RGB')
            return (self.pil2tensor(pil_image), )

    def 计算百分比像素值(self, 百分比, 遮罩宽, 遮罩高):
        """将百分比转换为像素值，基于遮罩白色区域宽高的平均值"""
        if 百分比 == 0:
            return 0
        平均值 = (遮罩宽 + 遮罩高) / 2
        像素值 = int(平均值 * 百分比 / 100)
        return max(1, 像素值)  # 确保至少为1像素，避免为0

    def 创建羽化遮罩(self, 遮罩, 扩展, 羽化):
        """浮点扩展与有限半径高斯羽化，核的尾部平滑归零。"""
        if 扩展 == 0 and 羽化 == 0:
            return 遮罩

        # 确保遮罩是单通道的numpy数组
        if 遮罩.ndim == 3:
            遮罩 = 遮罩[:, :, 0] if 遮罩.shape[2] > 1 else 遮罩[:, :, 0]

        # 从这里开始保持float32。遮罩如果先压成uint8再模糊，大半径时
        # 只有256个灰度级，容易在贴回后形成可见的波纹和断层。
        遮罩_float = np.clip(遮罩, 0.0, 1.0).astype(np.float32, copy=False)

        # 扩展遮罩。小半径保留原始椭圆膨胀；大半径使用距离变换，
        # 避免创建边长为(2*扩展+1)的大型结构元素。
        if 扩展 > 0:
            if 扩展 <= 64:
                核大小 = max(3, int(扩展) * 2 + 1)  # 确保是奇数
                核 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (核大小, 核大小))
                遮罩_float = cv2.dilate(遮罩_float, 核, iterations=1)
            else:
                # distanceTransform 的耗时主要与像素数量有关，基本不随
                # 扩展半径平方增长；使用非零像素作为原始遮罩区域。
                前景 = (遮罩_float > 1e-6).astype(np.uint8)
                到前景距离 = cv2.distanceTransform(1 - 前景, cv2.DIST_L2, 3)
                扩展区域 = 到前景距离 <= float(扩展)
                遮罩_float[扩展区域] = 1.0

        # 旧核在半径处仍有较大权重，硬截断后，透明区的第一步就会
        # 跳到非零值。对核的外半段平滑收尾，使过渡的两端斜率归零。
        if 羽化 > 0:
            羽化 = float(羽化)
            原高, 原宽 = 遮罩_float.shape[:2]
            缩小倍数 = max(1, int(math.ceil(羽化 / 64.0)))
            工作宽 = max(1, int(round(原宽 / 缩小倍数)))
            工作高 = max(1, int(round(原高 / 缩小倍数)))
            if 缩小倍数 > 1:
                工作遮罩 = cv2.resize(
                    遮罩_float, (工作宽, 工作高), interpolation=cv2.INTER_AREA
                )
            else:
                工作遮罩 = 遮罩_float

            核列表 = []
            for 工作半径 in (羽化 * 工作宽 / 原宽, 羽化 * 工作高 / 原高):
                核半径 = max(1, int(math.ceil(工作半径)))
                坐标 = np.arange(-核半径, 核半径 + 1, dtype=np.float64)
                # 核覆盖的是像素单元，半像素余量也让1像素羽化保留邻域权重。
                距离比例 = np.abs(坐标) / (工作半径 + 0.5)
                收尾 = np.clip((距离比例 - 0.5) * 2.0, 0.0, 1.0)
                平滑窗 = 1.0 - 收尾**3 * (收尾 * (收尾 * 6.0 - 15.0) + 10.0)
                核 = np.exp(-0.5 * (坐标 / 工作半径)**2) * 平滑窗
                核列表.append(核 / 核.sum())
            工作遮罩 = cv2.sepFilter2D(
                工作遮罩, -1, 核列表[0], 核列表[1], borderType=cv2.BORDER_REPLICATE
            )
            if 缩小倍数 > 1:
                遮罩_float = cv2.resize(
                    工作遮罩, (原宽, 原高), interpolation=cv2.INTER_LINEAR
                )
            else:
                遮罩_float = 工作遮罩

        # 限制数值范围，避免边界插值或浮点误差造成极小的越界值。
        return np.clip(遮罩_float, 0.0, 1.0).astype(np.float32, copy=False)

    def 缩放浮点遮罩(self, 遮罩, 宽, 高):
        """遮罩缩放不经过8位量化，也不使用有振铃的LANCZOS。"""
        遮罩 = np.clip(遮罩, 0.0, 1.0).astype(np.float32, copy=False)
        if 遮罩.shape[:2] == (高, 宽):
            return 遮罩
        插值 = cv2.INTER_AREA if 高 <= 遮罩.shape[0] and 宽 <= 遮罩.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(遮罩, (宽, 高), interpolation=插值)

    def 恢复遮罩到原始位置(self, 裁剪遮罩, 接缝):
        """将裁剪遮罩恢复到与未缩放裁剪遮罩相同的大小和位置"""
        # 获取接缝数据
        原始高, 原始宽 = 接缝["原像素"]
        裁剪顶部, 裁剪底部, 裁剪左侧, 裁剪右侧 = 接缝["裁剪区域"]
        目标高, 目标宽 = 接缝["目标尺寸"]
        输出尺寸模式 = 接缝.get("输出尺寸模式", "原像素")
        填充信息 = 接缝.get("填充信息", None)

        # 计算原始裁剪区域的尺寸
        原始裁剪高 = 裁剪底部 - 裁剪顶部
        原始裁剪宽 = 裁剪右侧 - 裁剪左侧

        # 将裁剪遮罩转换为numpy
        裁剪遮罩_np = 裁剪遮罩[0].cpu().numpy()
        处理高, 处理宽 = 裁剪遮罩_np.shape[:2]

        # 检查尺寸是否匹配
        if 处理高 != 目标高 or 处理宽 != 目标宽:
            # 如果不匹配，先缩放到目标尺寸
            裁剪遮罩_np = self.缩放浮点遮罩(裁剪遮罩_np, 目标宽, 目标高)

        # 对于自定义宽高模式，需要先恢复到扩展后的尺寸
        if 输出尺寸模式 == "自定义宽高":
            # 如果有填充信息，先去除填充
            if 填充信息 is not None:
                左填充, 上填充, 缩放宽, 缩放高, 目标宽, 目标高 = 填充信息
                # 从处理后的图像中提取有效区域（去除填充）
                有效区域 = 裁剪遮罩_np[上填充:上填充+缩放高, 左填充:左填充+缩放宽]
                裁剪遮罩_np = 有效区域
                处理高, 处理宽 = 裁剪遮罩_np.shape[:2]

        return self.缩放浮点遮罩(裁剪遮罩_np, 原始裁剪宽, 原始裁剪高)

    def 检查遮罩是否全黑(self, 遮罩):
        """检查遮罩是否全黑（几乎没有白色像素）"""
        if 遮罩 is None:
            return True
        # 计算非零像素的比例
        非零像素比例 = np.sum(遮罩 > 0.1) / 遮罩.size
        return 非零像素比例 < 0.001  # 如果非零像素比例小于0.1%，认为是全黑

    def 匹配裁剪颜色(self, 裁剪, 参考, 强度):
        """先排除明显重绘区域，再从轻微偏色的同位置像素估算统一校色。"""
        if 强度 <= 0:
            return 裁剪
        高, 宽 = 裁剪.shape[:2]
        比例 = min(1.0, 256.0 / max(高, 宽))
        工作尺寸 = (max(1, round(宽 * 比例)), max(1, round(高 * 比例)))
        源 = cv2.resize(裁剪, 工作尺寸, interpolation=cv2.INTER_AREA)
        目标 = cv2.resize(参考, 工作尺寸, interpolation=cv2.INTER_AREA)

        # 局部明暗结构基本不受整体偏色影响，降低新增/移除轮廓的采样权重。
        源灰度 = cv2.cvtColor(源, cv2.COLOR_RGB2GRAY)
        目标灰度 = cv2.cvtColor(目标, cv2.COLOR_RGB2GRAY)
        源细节 = 源灰度 - cv2.GaussianBlur(源灰度, (0, 0), 1.5)
        目标细节 = 目标灰度 - cv2.GaussianBlur(目标灰度, (0, 0), 1.5)
        结构权重 = 1.0 / (1.0 + ((源细节 - 目标细节) / 0.02)**2)
        # 仅接受轻微偏色：RGB各通道最多差24级，明度最多差18级（按0~255计）。
        候选区域 = (np.max(np.abs(目标 - 源), axis=2) <= 24.0 / 255.0)
        候选区域 &= (np.abs(目标灰度 - 源灰度) <= 18.0 / 255.0) & (结构权重 > 0.8)
        # 避开缩小图中混入主体颜色的边界，取样不依赖遮罩形状或大小。
        候选区域 = cv2.erode(候选区域.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        最少样本 = max(16, 候选区域.size * 0.05)
        if np.count_nonzero(候选区域) < 最少样本:
            return 裁剪
        源 = 源[候选区域].astype(np.float64)
        目标 = 目标[候选区域].astype(np.float64)
        结构权重 = 结构权重[候选区域]
        增益 = np.ones(3, dtype=np.float64)
        偏移 = np.median(目标 - 源, axis=0)
        for _ in range(6):
            残差 = 目标 - (源 * 增益 + 偏移)
            中心 = np.median(残差, axis=0)
            尺度 = np.maximum(np.median(np.abs(残差 - 中心), axis=0) * 1.4826, 0.004)
            偏差 = np.max(np.abs(残差 - 中心) / (4.685 * 尺度), axis=1)
            权重 = np.maximum(1.0 - 偏差**2, 0.0)**2 * 结构权重
            权重和 = 权重.sum()
            if 权重和 < max(16, len(源) * 0.1):
                return 裁剪
            权重 = 权重[:, None] / 权重和
            源均值 = (源 * 权重).sum(axis=0)
            目标均值 = (目标 * 权重).sum(axis=0)
            方差 = ((源 - 源均值)**2 * 权重).sum(axis=0)
            协方差 = ((源 - 源均值) * (目标 - 目标均值) * 权重).sum(axis=0)
            # 平坦背景缺少对比度信息，增益回归1，主要用偏移校正以免放大噪声。
            增益 = np.clip((协方差 + 1e-4) / (方差 + 1e-4), 0.9, 1.1)
            偏移 = np.clip(目标均值 - 源均值 * 增益, -24.0 / 255.0, 24.0 / 255.0)

        稳定区域 = (np.max(np.abs(目标 - (源 * 增益 + 偏移)), axis=1) < 0.035) & (结构权重 > 0.5)
        if np.count_nonzero(稳定区域) < max(最少样本, len(源) * 0.6):
            return 裁剪
        混合强度 = min(float(强度), 1.0)
        增益 = (1.0 + (增益 - 1.0) * 混合强度).astype(np.float32)
        偏移 = (偏移 * 混合强度).astype(np.float32)
        校正 = np.empty_like(裁剪)
        for 起点 in range(0, 高, 256):
            校正[起点:起点+256] = np.clip(裁剪[起点:起点+256] * 增益 + 偏移, 0.0, 1.0)
        return 校正

    def 恢复图像(self, 接缝, 裁剪图像, 遮罩扩展百分比, 遮罩羽化百分比, 裁剪遮罩=None, 背景图像=None, 颜色还原强度=0.0):
        # 添加RGB转换功能
        裁剪图像_rgb = self.image_to_rgb(裁剪图像)[0]
        if 背景图像 is not None:
            背景图像_rgb = self.image_to_rgb(背景图像)[0]
        else:
            背景图像_rgb = None

        # 获取接缝数据
        原始高, 原始宽 = 接缝["原像素"]
        裁剪顶部, 裁剪底部, 裁剪左侧, 裁剪右侧 = 接缝["裁剪区域"]
        目标高, 目标宽 = 接缝["目标尺寸"]
        缩放比例宽 = 接缝["缩放比例宽"]
        缩放比例高 = 接缝["缩放比例高"]
        最终缩放宽 = 接缝.get("最终缩放宽", 1.0)
        最终缩放高 = 接缝.get("最终缩放高", 1.0)
        输出尺寸模式 = 接缝.get("输出尺寸模式", "原像素")
        原始图像 = 接缝["原始图像"]
        遮罩填充 = 接缝.get("遮罩填充", False)
        遮罩白色区域宽 = 接缝.get("遮罩白色区域宽", 0)
        遮罩白色区域高 = 接缝.get("遮罩白色区域高", 0)
        填充信息 = 接缝.get("填充信息", None)
        未缩放裁剪遮罩 = 接缝.get("未缩放裁剪遮罩", None)

        # 使用填充后的遮罩（如果存在且开启了遮罩填充），否则使用原始遮罩
        if 遮罩填充 and "填充后遮罩" in 接缝:
            原始遮罩 = 接缝["填充后遮罩"]
        else:
            原始遮罩 = 接缝["原始遮罩"]

        # 将百分比转换为像素值
        遮罩扩展 = self.计算百分比像素值(遮罩扩展百分比, 遮罩白色区域宽, 遮罩白色区域高)
        遮罩羽化 = self.计算百分比像素值(遮罩羽化百分比, 遮罩白色区域宽, 遮罩白色区域高)

        # 获取背景图像
        if 背景图像_rgb is not None:
            # 检查背景图像尺寸是否匹配
            背景高, 背景宽 = 背景图像_rgb.shape[1:3]
            if 背景高 != 原始高 or 背景宽 != 原始宽:
                raise ValueError(f"背景图像尺寸({背景宽}x{背景高})与原始图像尺寸({原始宽}x{原始高})不匹配")
            背景 = 背景图像_rgb[0].cpu().numpy()
        else:
            # 使用接缝中的原始图像作为背景，也需要转换为RGB
            原始图像_rgb = self.image_to_rgb(原始图像)[0]
            背景 = 原始图像_rgb[0].cpu().numpy()

        # 转换裁剪图像为numpy
        裁剪_np = 裁剪图像_rgb[0].cpu().numpy()
        处理高, 处理宽 = 裁剪_np.shape[:2]

        强制匹配尺寸 = True  # 硬编码为True
        if 强制匹配尺寸 and (处理高 != 目标高 or 处理宽 != 目标宽):
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            pil裁剪 = pil裁剪.resize((目标宽, 目标高), Image.LANCZOS)
            裁剪_np = np.array(pil裁剪).astype(np.float32) / 255.0
            处理高, 处理宽 = 裁剪_np.shape[:2]


        # 检查尺寸是否匹配（如果强制匹配后仍不匹配，则使用原始逻辑）
        if 处理高 != 目标高 or 处理宽 != 目标宽:
            # 如果不匹配，先缩放到目标尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            pil裁剪 = pil裁剪.resize((目标宽, 目标高), Image.LANCZOS)
            裁剪_np = np.array(pil裁剪).astype(np.float32) / 255.0

        # 计算原始裁剪区域的尺寸
        原始裁剪高 = 裁剪底部 - 裁剪顶部
        原始裁剪宽 = 裁剪右侧 - 裁剪左侧

        # 对于自定义宽高模式，需要先恢复到扩展后的尺寸
        if 输出尺寸模式 == "自定义宽高":
            # 如果有填充信息，先去除填充
            if 填充信息 is not None:
                左填充, 上填充, 缩放宽, 缩放高, 目标宽, 目标高 = 填充信息
                # 从处理后的图像中提取有效区域（去除填充）
                有效区域 = 裁剪_np[上填充:上填充+缩放高, 左填充:左填充+缩放宽]
                裁剪_np = 有效区域
                处理高, 处理宽 = 裁剪_np.shape[:2]

            # 缩放到原始裁剪尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0
        else:
            # 其他模式直接缩放到原始裁剪尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0

        # 关键修改：优先使用传入的裁剪遮罩，如果存在且不是全黑
        if 裁剪遮罩 is not None and not self.检查遮罩是否全黑(裁剪遮罩[0].cpu().numpy()):
            # 使用传入的裁剪遮罩，先恢复到原始位置
            恢复遮罩_np = self.恢复遮罩到原始位置(裁剪遮罩, 接缝)
        else:
            # 使用未缩放的裁剪遮罩来计算遮罩位置
            if 未缩放裁剪遮罩 is not None:
                # 将未缩放的裁剪遮罩缩放到原始裁剪尺寸
                未缩放遮罩_np = np.array(未缩放裁剪遮罩).astype(np.float32) / 255.0
                恢复遮罩_np = self.缩放浮点遮罩(未缩放遮罩_np, 原始裁剪宽, 原始裁剪高)
            else:
                # 如果没有保存未缩放的裁剪遮罩，使用原始方法
                # 获取原始遮罩的对应区域
                原始遮罩_np = 原始遮罩[0].cpu().numpy()

                # 计算遮罩在原始图像中的有效区域
                遮罩顶部 = max(0, 裁剪顶部)
                遮罩底部 = min(原始高, 裁剪底部)
                遮罩左侧 = max(0, 裁剪左侧)
                遮罩右侧 = min(原始宽, 裁剪右侧)

                遮罩区域 = 原始遮罩_np[遮罩顶部:遮罩底部, 遮罩左侧:遮罩右侧]

                # 缩放遮罩到恢复后的尺寸
                if 遮罩区域.size > 0:
                    恢复遮罩_np = self.缩放浮点遮罩(遮罩区域, 原始裁剪宽, 原始裁剪高)
                else:
                    恢复遮罩_np = np.zeros((原始裁剪高, 原始裁剪宽), dtype=np.float32)

        # 只在画布内计算羽化，画布外的黑色填充不应使贴边遮罩变灰。
        有效左 = max(0, -裁剪左侧)
        有效上 = max(0, -裁剪顶部)
        有效右 = min(原始裁剪宽, 原始宽 - 裁剪左侧)
        有效下 = min(原始裁剪高, 原始高 - 裁剪顶部)
        羽化遮罩 = np.zeros_like(恢复遮罩_np, dtype=np.float32)
        if 有效右 > 有效左 and 有效下 > 有效上:
            有效遮罩 = self.创建羽化遮罩(
                恢复遮罩_np[有效上:有效下, 有效左:有效右], 遮罩扩展, 遮罩羽化
            )
            # 贴回区域之外没有处理后的图像可融合；在这些人工边界内
            # 平滑降至0。与原图画布重合的边则无需额外淡出。
            if 遮罩羽化 > 0:
                for 轴, 低端, 高端 in (
                    (0, 裁剪顶部 > 0, 裁剪底部 < 原始高),
                    (1, 裁剪左侧 > 0, 裁剪右侧 < 原始宽),
                ):
                    长度 = 有效遮罩.shape[轴]
                    位置 = np.arange(长度, dtype=np.float32)
                    权重 = np.ones(长度, dtype=np.float32)
                    if 低端:
                        权重 = np.minimum(权重, 位置 / 遮罩羽化)
                    if 高端:
                        权重 = np.minimum(权重, (长度 - 1 - 位置) / 遮罩羽化)
                    权重 = np.clip(权重, 0.0, 1.0)
                    权重 = 权重**3 * (权重 * (权重 * 6.0 - 15.0) + 10.0)
                    有效遮罩 *= 权重[:, None] if 轴 == 0 else 权重[None, :]
            羽化遮罩[有效上:有效下, 有效左:有效右] = 有效遮罩

        # 确保遮罩是3通道的
        if 恢复后_np.ndim == 3:
            遮罩3d = np.stack([羽化遮罩, 羽化遮罩, 羽化遮罩], axis=-1)
        else:
            遮罩3d = 羽化遮罩

        # 将裁剪图像融合到背景中
        结果 = 背景.copy()

        # 计算粘贴位置（只粘贴在原始图像内的部分）
        粘贴x = max(0, 裁剪左侧)
        粘贴y = max(0, 裁剪顶部)

        # 计算在恢复图像中对应的区域
        源x偏移 = max(0, -裁剪左侧)
        源y偏移 = max(0, -裁剪顶部)
        源宽 = min(原始裁剪宽 - 源x偏移, 原始宽 - 粘贴x)
        源高 = min(原始裁剪高 - 源y偏移, 原始高 - 粘贴y)

        if 源宽 > 0 and 源高 > 0:
            # 提取要粘贴的区域
            粘贴区域 = 结果[粘贴y:粘贴y+源高, 粘贴x:粘贴x+源宽]

            # 提取对应的恢复图像区域
            恢复区域 = 恢复后_np[源y偏移:源y偏移+源高, 源x偏移:源x偏移+源宽]

            if 颜色还原强度 > 0:
                # 始终参考原图同位置区域，传入的替换背景不参与校色估计。
                参考区域 = 原始图像[0, 粘贴y:粘贴y+源高, 粘贴x:粘贴x+源宽, :3].cpu().numpy()
                恢复区域 = self.匹配裁剪颜色(恢复区域, 参考区域, 颜色还原强度)

            # 提取对应的遮罩区域
            遮罩区域 = 遮罩3d[源y偏移:源y偏移+源高, 源x偏移:源x偏移+源宽]

            # 使用遮罩进行融合
            融合区域 = 粘贴区域 * (1 - 遮罩区域) + 恢复区域 * 遮罩区域
            if 遮罩羽化 > 0:
                # PreviewImage/SaveImage直接截断到uint8，极小的高斯尾部也会
                # 让整圈像素少一级灰度。在混合区加入不足1/255的确定性抖动，
                # 分散量化台阶；限制在两张源图之间，避免产生亮边或改变底图。
                横坐标 = np.arange(粘贴x, 粘贴x + 源宽, dtype=np.float64)[None, :]
                for 行起点 in range(0, 源高, 128):
                    行终点 = min(行起点 + 128, 源高)
                    当前遮罩 = 遮罩区域[行起点:行终点]
                    过渡区域 = (当前遮罩 > 0.0) & (当前遮罩 < 1.0)
                    if not np.any(过渡区域):
                        continue
                    纵坐标 = np.arange(粘贴y + 行起点, 粘贴y + 行终点, dtype=np.float64)[:, None]
                    抖动 = np.mod(52.9829189 * np.mod(横坐标 * 0.06711056 + 纵坐标 * 0.00583715, 1.0), 1.0)
                    当前融合 = 融合区域[行起点:行终点]
                    上限 = np.maximum(粘贴区域[行起点:行终点], 恢复区域[行起点:行终点])
                    np.minimum(当前融合 + (抖动 / 255.0).astype(np.float32)[..., None], 上限,
                               out=当前融合, where=过渡区域)
            结果[粘贴y:粘贴y+源高, 粘贴x:粘贴x+源宽] = 融合区域

        # 转换回tensor
        结果_tensor = torch.from_numpy(结果)[None, ...]

        return (结果_tensor,)

Restore = 孤海_裁剪恢复
