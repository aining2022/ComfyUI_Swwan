"""
Fast Groups Muter Node - 移植自 rgthree-comfy
快速静音/取消静音工作流中的组内所有节点
这是一个纯前端虚拟节点，主要逻辑在 web/js/fast_groups_muter.js 中实现
"""

class FastGroupsMuter:
    """虚拟节点，用于快速控制工作流中组内节点的静音状态"""
    
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


NODE_CLASS_MAPPINGS = {
    "Fast Groups Muter (rgthree)": FastGroupsMuter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fast Groups Muter (rgthree)": "Fast Groups Muter"
}
