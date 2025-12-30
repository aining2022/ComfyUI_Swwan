class AnyType(str):
    """A special type that compares equal to any other type."""
    def __ne__(self, __value: object) -> bool:
        return False

    def __eq__(self, __value: object) -> bool:
        return True

    def __str__(self):
        return "*"

ANY = AnyType("*")

class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "优先输入": (ANY,),
                "备用输入": (ANY,)
            }
        }

    RETURN_TYPES = ("BOOLEAN", ANY)
    RETURN_NAMES = ("是否启用优先", "输出结果")
    FUNCTION = "check"
    CATEGORY = "Swwan/utils"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check(self, 优先输入=None, 备用输入=None):
        # 如果优先输入不为空，则认为激活（使用优先输入）
        is_active = 优先输入 is not None
        output_data = 优先输入 if is_active else 备用输入
        return (is_active, output_data)

class AnyBooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入": (ANY,),
                "开关": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("输出结果",)
    FUNCTION = "process"
    CATEGORY = "Swwan/utils"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(self, 开关, 输入=None):
        if 开关:
            return (输入,)
        else:
            return (None,)

NODE_CLASS_MAPPINGS = {
    "AnySwitch (Swwan)": AnySwitch,
    "AnyBooleanSwitch (Swwan)": AnyBooleanSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnySwitch (Swwan)": "Any Switch (Swwan)",
    "AnyBooleanSwitch (Swwan)": "Any Boolean Switch (Swwan)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
