# SPDX-License-Identifier: MIT
# Apt_Preset source retained; see licenses/MIT-Apt.txt and THIRD_PARTY_NOTICES.md.
import math
import torch
import comfy
import numpy as np
from typing import Any, Callable, Mapping
from nodes import NODE_CLASS_MAPPINGS
from ..ops.types import ANY_TYPE, any_type
from ..ops.easing import EASING_TYPES, apply_easing, easing_functions
from ..ops.workflow_helpers import get_input_nodes, get_input_types, keyframe_scheduler, prompt_scheduler


class math_Remap_data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "easing": (EASING_TYPES, {"default": "Linear"}),
            },
            "optional": {
                "value": (ANY_TYPE,),  # 移到optional中，变为可选项
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("float", "int",)
    CATEGORY = "Apt_Preset/data"

    def set_range(
        self,
        clamp,
        source_min,
        source_max,
        target_min,
        target_max,
        easing,
        value=None,  # 设为可选参数，默认值为None
    ):
        # 处理value为None的情况（未输入时），这里默认设为0.0，可根据需求调整
        if value is None:
            float_value = 0.0
        else:
            try:
                float_value = float(value)
            except ValueError:
                raise ValueError("Invalid value for conversion to float")

        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (float_value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)
        eased_value = apply_easing(normalized_value, easing)
        if clamp:
           eased_value = max(min(eased_value, 1), 0)
        res_float = target_min + (target_max - target_min) * eased_value
        res_int = int(res_float)

        return (res_float, res_int)

class math_calculate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 定义预设运算列表（使用扁平化命名，避免斜杠等特殊符号）
        presets = [
            # 单值运算
            ("custom", "自定义表达式"),
            ("sin(a)", "正弦函数 sin(a)"),
            ("cos(a)", "余弦函数 cos(a)"),
            ("tan(a)", "正切函数 tan(a)"),
            ("asin(a)", "反正弦函数 asin(a)"),
            ("acos(a)", "反余弦函数 acos(a)"),
            ("atan(a)", "反正切函数 atan(a)"),
            ("pow(a, 2)", "平方 a²"),
            ("sqrt(a)", "平方根 √a"),
            ("log(a)", "自然对数 log(a)"),
            ("log10(a)", "常用对数 log10(a)"),
            ("sinh(a)", "双曲正弦 sinh(a)"),
            ("cosh(a)", "双曲余弦 cosh(a)"),
            ("tanh(a)", "双曲正切 tanh(a)"),
            ("asinh(a)", "反双曲正弦 asinh(a)"),
            ("acosh(a)", "反双曲余弦 acosh(a)"),
            ("atanh(a)", "反双曲正切 atanh(a)"),
            ("radians(a)", "角度转弧度 radians(a)"),
            ("degrees(a)", "弧度转角度 degrees(a)"),
            ("fabs(a)", "绝对值 fabs(a)"),
            ("exp(a)", "指数函数 e的a次方"),
            ("round(a)", "四舍五入 round(a)"),
            ("ceil(a)", "向上取整 ceil(a)"),
            ("floor(a)", "向下取整 floor(a)"),
            ("abs(a)", "绝对值 abs(a)"),

            # 双值运算
            ("a + b", "加法 a + b"),
            ("a - b", "减法 a - b"),
            ("a * b", "乘法 a * b"),
            ("a ÷ b", "除法 a 除以 b"),
            ("a % b", "取模 a 模 b"),
            ("pow(a,b)", "幂运算 a的b次方"),
            ("ceil(a÷b)", "向上取整 ceil(a÷b)"),
            ("floor(a÷b)", "向下取整 floor(a÷b)"),
            ("max(a,b)", "最大值 max(a,b)"),
            ("min(a,b)", "最小值 min(a,b)"),
            ("a > b", "大于 a > b"),
            ("a < b", "小于 a < b"),
            ("a >= b", "大于等于 a >= b"),
            ("a <= b", "小于等于 a <= b"),
            ("a == b", "等于 a == b"),
            ("a != b", "不等于 a != b"),
            ("a & b", "按位与 a & b"),
            ("a | b", "按位或 a | b"),
            ("a ^ b", "按位异或 a ^ b"),
            ("a << b", "左移位 a << b"),
            ("a >> b", "右移位 a >> b"),
            ("atan2(a,b)", "四象限反正切 atan2(a,b)"),
            ("hypot(a,b)", "直角三角形斜边 hypot(a,b)"),
            ("copysign(a,b)", "复制符号 copysign(a,b)"),
            ("fmod(a,b)", "浮点数取模 fmod(a,b)"),

            # 三值运算（仅保留三个值的最大值和最小值）
            ("max(a,b,c)", "最大值 max(a,b,c)"),
            ("min(a,b,c)", "最小值 min(a,b,c)"),
            ("clamp(a,b,c)", "限制在b和c之间 clamp(a,b,c)"),
            ("lerp(a,b,c)", "线性插值 lerp(a,b,c)"),
        ]

        return {
            "required": {
                "preset": (
                    [p[0] for p in presets],
                    {"default": "custom", "label": [p[1] for p in presets]}
                ),
                "expression": ("STRING", {"default": "", "multiline": False,}),
                "a": (ANY_TYPE, {"forceInput": True}),
            },
            "optional": {
                "b": (ANY_TYPE,),
                "c": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("float_result", "int_result", "bool_result")
    FUNCTION = "calculate"
    CATEGORY = "Apt_Preset/data"
    DESCRIPTION = """
    - 基本运算：加 (+)、减 (-)、乘 (*)、除 (/)、模 (%)
    - 三角函数: sin(a)、cos、tan、asin、acos、atan、atan2(a,b)
    - 幂运算与开方: pow(a,2)=a*a、sqrt、hypot(a,b)
    - 对数运算: log、log10(a)、exp(a)
    - 双曲函数: sinh、cosh、tanh、asinh、acosh、atanh
    - 角度与弧度转换: radians、degrees
    - 绝对值与取整: fabs、abs、ceil、floor、round、sign
    - 位运算: &(与)、|(或)、^(异或)、<<(左移)、>>(右移)
    - 比大小: max(a,b,c) ,min(a,b,c)
    - 布尔运算: a>b,a<b,a>=b,a<=b,a==b,a!=b ,返回True或False
    - 其他运算: clamp(a,b,c)、lerp(a,b,c)、if(a,b,c)、copysign(a,b)、fmod(a,b)
    """


    def calculate(self, preset, expression, a, b=None, c=None):
        from ..ops.expressions import evaluate_expression
        try:
            expr = expression if preset == "custom" else preset
            result = evaluate_expression(expr, {"a": a, "b": b if b is not None else 0, "c": c if c is not None else 0})
            return (float(result), int(result), bool(result))
        except (ValueError, TypeError, SyntaxError, ArithmeticError):
            return (0.0, 0, False)
