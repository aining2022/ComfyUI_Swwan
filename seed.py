"""
Seed Node - 移植自 rgthree-comfy
增强的种子节点，支持随机、递增、递减等特殊功能
"""

import random
from datetime import datetime

# 初始化独立的随机状态
initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
rgthree_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def new_random_seed():
    """从 rgthree_seed_random_state 获取新的随机种子"""
    global rgthree_seed_random_state
    prev_random_state = random.getstate()
    random.setstate(rgthree_seed_random_state)
    seed = random.randint(1, 1125899906842624)
    rgthree_seed_random_state = random.getstate()
    random.setstate(prev_random_state)
    return seed


class RgthreeSeed:
    """增强的种子节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": -1125899906842624, "max": 1125899906842624},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "main"
    CATEGORY = "rgthree"

    @classmethod
    def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None):
        """如果是特殊种子值，强制标记为已更改"""
        if seed in (-1, -2, -3):
            return new_random_seed()
        return seed

    def main(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
        """返回种子值"""
        # -1: 随机, -2: 递增, -3: 递减
        # 这些特殊值通常在前端处理，但如果直接通过 API 调用，这里会生成随机种子
        if seed in (-1, -2, -3):
            print(
                f"[Seed] 警告: 收到特殊种子值 {seed}，这通常不应该发生。生成随机种子。"
            )
            original_seed = seed
            seed = new_random_seed()
            print(f"[Seed] 服务器生成随机种子: {seed}")

            # 尝试保存到工作流元数据
            if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
                workflow_node = next(
                    (
                        x
                        for x in extra_pnginfo["workflow"]["nodes"]
                        if str(x["id"]) == str(unique_id)
                    ),
                    None,
                )
                if workflow_node and "widgets_values" in workflow_node:
                    for index, widget_value in enumerate(
                        workflow_node["widgets_values"]
                    ):
                        if widget_value == original_seed:
                            workflow_node["widgets_values"][index] = seed

            # 保存到 prompt 元数据
            if unique_id and prompt and str(unique_id) in prompt:
                prompt_node = prompt[str(unique_id)]
                if "inputs" in prompt_node and "seed" in prompt_node["inputs"]:
                    prompt_node["inputs"]["seed"] = seed

        return (seed,)


NODE_CLASS_MAPPINGS = {"Seed (rgthree)": RgthreeSeed}

NODE_DISPLAY_NAME_MAPPINGS = {"Seed (rgthree)": "Seed"}
