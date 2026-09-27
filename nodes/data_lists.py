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


class list_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": -1, "min": -9007199254740991}),
            }
        }

    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )  # 确保输出是列表形式
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

    def run(self, ANY: list, start: list[int], end: list[int]):
        # 从输入列表中获取起始和结束值
        start_val = start[0] if start else 0
        end_val = end[0] if end else -1

        # 处理负数索引
        if start_val < 0:
            start_val = len(ANY) + start_val
        if end_val < 0:
            end_val = len(ANY) + end_val

        # 确保索引在有效范围内
        start_val = max(0, min(start_val, len(ANY)))
        end_val = max(0, min(end_val, len(ANY)))

        # 确保start不大于end
        if start_val > end_val:
            return ([], )

        # 执行切片操作
        sliced = ANY[start_val:end_val]
        return (sliced, )

class list_Merge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    NAME = "list_Merge"
    INPUT_IS_LIST = True
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        unique_id = unique_id[0]
        prompt = prompt[0]
        extra_pnginfo = extra_pnginfo[0]
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )

class list_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                             "max_frames": ("INT", {"default": 100, "min": 1, "max": 99999}),  # 添加 max_frames 参数
                             "easing_type": (list(easing_functions.keys()), ),
                },
        }
    RETURN_TYPES = ("FLOAT","INT",  "FLOAT")
    RETURN_NAMES = ("float","int",  "weight")
    OUTPUT_IS_LIST = (True, True, True)  # 强制输出结果为列表
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data"

    def adv_schedule(self, schedule, max_frames, easing_type):
        schedule_lines = list()
        if schedule == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
            return ([], [], [])  # 返回空列表

        lines = schedule.split('\n')
        for line in lines:
            schedule_lines.extend([("ADV", line)])

        int_out_list = []
        value_out_list = []
        weight_list = []

        for current_frame in range(max_frames):
            params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
            if params == "":
                print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame {current_frame}. Advanced schedules must start at frame 0.")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)
                continue

            try:
                current_params, next_params, from_index, to_index = prompt_scheduler(schedule_lines, "ADV", current_frame)
                if to_index == from_index:
                    t = 1.0
                else:
                    t = (current_frame - from_index) / (to_index - from_index)
                if t < 0 or t > 1:
                    t = 1.0
                weight = apply_easing(t, easing_type)
                current_value = float(current_params)
                next_value = float(next_params)
                value_out = current_value + (next_value - current_value) * weight
                int_out = int(value_out)

                int_out_list.append(int_out)
                value_out_list.append(value_out)
                weight_list.append(weight)
            except ValueError:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)

        return ( value_out_list, int_out_list, weight_list)

class list_num_range:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "num": ("INT", {"default": 10, "min": 2}),
            },
        }

    TITLE = "Create Linspace"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("data", "list", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

    def run(self, start: float, stop: float, num: int):
        range_list = list(np.linspace(start, stop, num))
        return (range_list, range_list, len(range_list))

class BatchSlice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": -1, "min": -9007199254740991}),  # 默认-1表示到末尾
            }
        }

    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("Data", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

    def run(self, LIST: list, start: int, end: int):
        list_length = len(LIST)

        # 处理负数索引
        if start < 0:
            start = list_length + start
        if end < 0:
            end = list_length + end

        # 确保索引在有效范围内
        start = max(0, min(start, list_length))
        end = max(0, min(end, list_length))

        # 确保start不大于end
        if start > end:
            # 返回空列表或适当的默认值
            # 检查输入数据类型以返回相应类型的空值
            if list_length > 0 and isinstance(LIST[0], torch.Tensor):
                # 如果是张量列表，返回空的张量
                return (torch.tensor([]), )
            return ([], )

        # 执行切片操作
        sliced_data = LIST[start:end]

        # 如果列表中的元素是张量，考虑将它们堆叠成一个张量
        if len(sliced_data) > 0 and isinstance(sliced_data[0], torch.Tensor):
            try:
                # 如果是相同形状的张量，尝试堆叠它们
                return (torch.stack(sliced_data), )
            except RuntimeError:
                # 如果形状不匹配，返回原列表
                return (sliced_data, )

        return (sliced_data, )

class MergeBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    NAME = "list_MergeBatch"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("list", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )

class type_AnyIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            },
            "hidden":{
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    INPUT_IS_LIST = True
    FUNCTION = "getIndex"
    CATEGORY = "Apt_Preset/data"

    def getIndex(self, any, index, prompt=None, my_unique_id=None):
        index = index[0]
        prompt = prompt[0]
        my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split('.')[len(my_unique_id.split('.')) - 1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]['inputs']['any']
        class_type = prompt[id]['class_type']
        node_class = NODE_CLASS_MAPPINGS [class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, 'OUTPUT_IS_LIST') else False

        if output_is_list or len(any) > 1:
            return (any[index],)
        elif isinstance(any[0], torch.Tensor):
            batch_index = min(any[0].shape[0] - 1, index)
            s = any[0][index:index + 1].clone()
            return (s,)
        else:
            return (any[0][index],)
