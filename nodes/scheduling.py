# SPDX-License-Identifier: MIT
# Scheduling ancestry: Comfyroll Studio (RockOfFire / Akatsuzi); received via Apt_Preset.
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


class sch_split_text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "a b c"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
            },
            "optional": {
                "preset": (
                    ["None", "Line", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe", "Custom"],
                    {"default": "None"}
                ),
                "delimiter": (
                    "STRING",
                    {"default": " ", "label": "Custom delimiter", }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ('i_text', "length",)
    FUNCTION = "text_to_list"
    CATEGORY = "Apt_Preset/data/schedule"
    DESCRIPTION = """
    文本拆分预设说明
    - **None**：不采用预设分隔符。
    - **Line**：以`\n`或`\r\n`（换行符）拆分。
    - **Space**：用` `（空格）或`　`（全角空格）拆分。
    - **Comma**：以`,`（逗号）或`，`（中文逗号）拆分。
    - **Period**：用`.`（句号）或`。`（中文句号）拆分。
    - **Semicolon**：以`;`（分号）或`；`（中文分号）拆分。
    - **Tab**：用`\t`（制表符）拆分。
    - **Pipe**：以`|`（竖线）拆分。
    - **Custom**：使用自定义分隔符（支持转义字符如`\\n`、`\\t`）。
    """

    def text_to_list(self, text, current_frame, preset="None", delimiter=" "):
        preset_map = {
            "None": [],
            "Line": ["\n", "\r\n"],
            "Space": [" ", "　"],
            "Comma": [",", "，"],
            "Period": [".", "。"],
            "Semicolon": [";", "；"],
            "Tab": ["\t"],
            "Pipe": ["|"],
            "Custom": []
        }
        separators = preset_map.get(preset, [])

        if (preset == "Custom" or (preset == "None" and delimiter)) and delimiter:
            delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            separators = [delimiter]

        if text.strip() == "":
            strList = []
        elif not separators:
            strList = [text.strip()] if text.strip() else []
        else:
            escaped_seps = [re.escape(sep) for sep in separators if sep]
            sep_pattern = '|'.join(escaped_seps)
            strList = re.split(f'(?:{sep_pattern})', text.strip())
            strList = [item.strip() for item in strList if item.strip()]

        list_length = len(strList)
        if current_frame < 0 or current_frame >= list_length:
            return ("", list_length)

        return (strList[current_frame], list_length)

class sch_text:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                },
                "optional": {
                }

        }
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("current_prompt", "next_prompt", "weight")
    FUNCTION = "simple_schedule"
    CATEGORY = "Apt_Preset/data/schedule"

    def simple_schedule(self, keyframe_list,  current_frame, easing_type,):
        keyframes = list()
        if keyframe_list == "":
            print(f"[Error] CR Simple Prompt Scheduler. No lines in keyframe list")
            return ()
        lines = keyframe_list.split('\n')
        for line in lines:
            if not line.strip():
                print(f"[Warning] CR Simple Prompt Scheduler. Skipped blank line at line {i}")
                continue
            keyframes.extend([("SIMPLE", line)])
        current_prompt, next_prompt, current_keyframe, next_keyframe = prompt_scheduler(keyframes, "SIMPLE", current_frame)
        if current_prompt == "":
            print(f"[Warning] CR Simple Prompt Scheduler. No prompt found for frame. Simple schedules must start at frame 0.")
        else:
            try:
                current_prompt_out = str(current_prompt)
                next_prompt_out = str(next_prompt)
                from_index = int(current_keyframe)
                to_index = int(next_keyframe)
            except ValueError:
                print(f"[Warning] CR Simple Text Scheduler. Invalid keyframe at frame {current_frame}")

            if from_index == to_index:
                weight_out = 1.0
            else:
                # 缓入缓出效果
                t = (to_index - current_frame) / (to_index - from_index)

                weight_out =  apply_easing(t, easing_type)


            return(current_prompt_out, next_prompt_out, weight_out)

class sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("INT", "FLOAT", "weight", "length")
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data/schedule"

    def adv_schedule(self, schedule, current_frame, easing_type):
        int_out = 0
        value_out = 0.0
        weight = 0.0
        data_length = 0

        if schedule.strip() == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
        else:
            lines = [line.strip() for line in schedule.split('\n') if line.strip()]
            frame_numbers = []
            for line in lines:
                if '@' in line:
                    try:
                        frame_part = line.split('@')[0].strip()
                        frame = int(frame_part)
                        frame_numbers.append(frame)
                    except ValueError:
                        print(f"[Warning] CR Advanced Value Scheduler. Invalid frame number: {frame_part} in line: {line}")

            if frame_numbers:
                min_frame = min(frame_numbers)
                max_frame = max(frame_numbers)
                data_length = max_frame - min_frame + 1
                print(f"[Info] CR Advanced Value Scheduler. Actual data length (frame range): {data_length} frames (from {min_frame} to {max_frame})")
            else:
                print(f"[Warning] CR Advanced Value Scheduler. No valid frame numbers found in schedule")

        schedule_lines = list()
        if schedule == "":
            return (int_out, value_out, weight, data_length)

        lines = schedule.split('\n')
        for line in lines:
            if line.strip():
                schedule_lines.extend([("ADV", line)])

        params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
        if params == "":
            print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame. Advanced schedules must start at frame 0.")
        else:
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

                print(f"[Info] CR Advanced Value Scheduler. Processing frame {current_frame}, current data index range: {from_index} -> {to_index}")

            except ValueError as e:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}: {str(e)}")

        return (int_out, value_out, weight, data_length)

class sch_Prompt:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",),
                            "keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("CONDITIONING", )
    FUNCTION = "condition"
    CATEGORY = "Apt_Preset/data/schedule"

    def condition(self, clip, keyframe_list, current_frame, easing_type):

        (current_prompt, next_prompt, weight) = sch_text().simple_schedule( keyframe_list, current_frame, easing_type)

        # CLIP text encoding
        tokens = clip.tokenize(str(next_prompt))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(str(current_prompt))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
        print(weight)

        # Average conditioning
        conditioning_to_strength = weight
        conditioning_from = [[cond_from, {"pooled_output": pooled_from}]]
        conditioning_to = [[cond_to, {"pooled_output": pooled_to}]]
        out = []

        if len(conditioning_from) > 1:
            print("Warning: Conditioning from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)

        return (out,)

class sch_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_image",)
    FUNCTION = "select_image"
    CATEGORY = "Apt_Preset/data/schedule"

    def select_image(self, images, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(images) - 1)  # 调整当前帧
        selected_image = images[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or image count. Using frame {adjusted_frame}.")
        return (selected_image,)

class sch_mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("selected_mask",)
    FUNCTION = "select_mask"
    CATEGORY = "Apt_Preset/data/schedule"

    def select_mask(self, masks, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(masks) - 1)  # 调整当前帧
        selected_mask = masks[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or mask count. Using frame {adjusted_frame}.")
        return (selected_mask,)
