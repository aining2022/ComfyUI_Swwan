"""
Fast Muter Node - 移植自 rgthree-comfy
快速静音/取消静音连接的节点
这是一个虚拟节点，主要逻辑在 web/js 中实现
"""


class FastMuter:
    """虚拟节点，用于快速控制连接节点的静音状态"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("OPT_CONNECTION",)
    FUNCTION = "execute"
    CATEGORY = "rgthree"
    OUTPUT_NODE = False

    def execute(self):
        return (None,)


NODE_CLASS_MAPPINGS = {"Fast Muter (rgthree)": FastMuter}

NODE_DISPLAY_NAME_MAPPINGS = {"Fast Muter (rgthree)": "Fast Muter"}
