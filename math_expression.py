# SPDX-License-Identifier: MIT
# Source: UniversalToolkit; original notices and modification history in THIRD_PARTY_NOTICES.md.
from .ops.expressions import evaluate_expression
from .ops.math_presets import PRESETS


# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")

# Completion hints only; evaluation and allowed functions live in ops/expressions.py.
functions = {name: {'hint': hint} for name, hint in {
    'round': 'number, dp? = 0', 'ceil': 'number', 'floor': 'number',
    'min': '...numbers', 'max': '...numbers', 'randomint': 'min, max',
    'randomchoice': '...numbers', 'sqrt': 'number', 'int': 'number',
    'iif': 'value, truepart, falsepart',
}.items()}

autocompleteWords = list(
    {
        "text": x,
        "value": f"{x}()",
        "showValue": False,
        "hint": f"{functions[x]['hint']}",
        "caretOffset": -1,
    }
    for x in functions.keys()
)


class MathExpression_UTK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "pysssss.autocomplete": {
                            "words": autocompleteWords,
                            "separator": "",
                        },
                    },
                ),
            },
            "optional": {
                "a": (any,),
                "b": (any,),
                "c": (any,),
                "preset": ([p[0] for p in PRESETS], {"default": "custom"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
        "BOOLEAN",
    )
    FUNCTION = "evaluate"
    CATEGORY = "Swwan/Math"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(s, expression, preset="custom", **kwargs):
        expression = expression if preset == "custom" else preset
        if "random" in expression:
            return float("nan")
        return expression

    def get_widget_value(self, extra_pnginfo, prompt, node_name, widget_name):
        workflow = (
            extra_pnginfo["workflow"] if "workflow" in extra_pnginfo else {"nodes": []}
        )
        node_id = None
        for node in workflow["nodes"]:
            name = node["type"]
            if "properties" in node:
                if "Node name for S&R" in node["properties"]:
                    name = node["properties"]["Node name for S&R"]
            if name == node_name:
                node_id = node["id"]
                break
            if "title" in node:
                name = node["title"]
            if name == node_name:
                node_id = node["id"]
                break
        if node_id is not None:
            values = prompt[str(node_id)]
            if "inputs" in values:
                if widget_name in values["inputs"]:
                    value = values["inputs"][widget_name]
                    if isinstance(value, list):
                        raise ValueError(
                            "Converted widgets are not supported via named reference, use the inputs instead."
                        )
                    return value
            raise NameError(f"Widget not found: {node_name}.{widget_name}")
        raise NameError(f"Node not found: {node_name}.{widget_name}")

    def get_size(self, target, property):
        if isinstance(target, dict) and "samples" in target:
            # Latent
            if property == "width":
                return target["samples"].shape[3] * 8
            return target["samples"].shape[2] * 8
        else:
            # Image
            if property == "width":
                return target.shape[2]
            return target.shape[1]

    def evaluate(self, expression, prompt=None, extra_pnginfo=None, a=None, b=None, c=None, preset="custom"):
        def attribute(name, field):
            if name in {"a", "b", "c"} and field in {"width", "height"}:
                return self.get_size({"a": a, "b": b, "c": c}[name], field)
            return self.get_widget_value(extra_pnginfo or {}, prompt or {}, name, field)
        expr = expression if preset == "custom" else preset
        result = evaluate_expression(expr, {"a": a, "b": b, "c": c}, attribute, legacy_ast=preset == "custom")
        return {"ui": {"value": [result]}, "result": (int(result), float(result), bool(result))}
