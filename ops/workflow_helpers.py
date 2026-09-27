# SPDX-License-Identifier: MIT
# Scheduling ancestry: Comfyroll Studio (RockOfFire / Akatsuzi); received via Apt_Preset.
# Apt_Preset source retained; see licenses/MIT-Apt.txt and THIRD_PARTY_NOTICES.md.
from nodes import NODE_CLASS_MAPPINGS


def get_input_nodes(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_nodes = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_nodes.append(in_node)
    return input_nodes

def get_input_types(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_types = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_type = in_node["outputs"][in_socket_id]["type"]
        input_types.append(input_type)
    return input_types

def keyframe_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_params = ""
    for item in schedule:
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        if not line.strip():
            print(f"[Warning] Skipped blank line at line {i}")
            continue
        frame_str, params = line.split('@', 1)
        frame = int(frame_str)
        params = params.lstrip()
        if frame < current_frame:
            previous_params = params
            continue
        if frame == current_frame:
            previous_params = params
        else:
            params = previous_params
        return params
    return previous_params

def prompt_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_prompt = ""
    previous_keyframe = 0
    for item in schedule:
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        frame_str, prompt = line.split('@', 1)
        frame_str = frame_str.strip('\"')
        frame = int(frame_str)
        prompt = prompt.lstrip()
        prompt = prompt.replace('"', '')
        if frame < current_frame:
            previous_prompt = prompt
            previous_keyframe = frame
            continue
        elif frame == current_frame:
            next_prompt = prompt
            next_keyframe = frame
            previous_prompt = prompt
            previous_keyframe = frame
        else:
            next_prompt = prompt
            next_keyframe = frame
            prompt = previous_prompt
        return prompt, next_prompt, previous_keyframe, next_keyframe
    return previous_prompt, previous_prompt, previous_keyframe, previous_keyframe
