"""Migrate the supplied UI workflow using the installed Swwan node schemas.

Usage (inside a ComfyUI Python environment):
    python scripts/migrate_qwen2511_workflow.py INPUT OUTPUT --comfyui-root PATH
The input file is never modified. No model inference or download is performed.
"""
import argparse
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = {
    "GH_MaskCropV2": "SwwanCropByMaskV5",
    "GH_CropRestore": "SwwanRestoreCropBoxV4",
    "图像缩放V2_孤海": "ImageResizeKJv2Alternative",
    "GulfSeaImageMergeMask": "ImageBlendSwwan",
    "图像缩放范围孤海": "SwwanImageResizeRange",
    "BlockifyMask": "SwwanBlockifyMask",
    "Images to RGB": "SwwanImagesToRGB",
    "ColorConverterGuhai": "SwwanColorConverter",
    "LayerUtility: ColorImage": "LayerUtility: ColorImage (Swwan)",
}
INPUT_NAMES = {
    "GH_MaskCropV2": {"图像": "image", "遮罩": "mask", "遮罩填充": "fill_mask_holes", "扩展系数上": "top_reserve_ratio", "扩展系数下": "bottom_reserve_ratio", "扩展系数左": "left_reserve_ratio", "扩展系数右": "right_reserve_ratio", "输出尺寸": "output_size", "自定义宽": "custom_width", "自定义高": "custom_height", "倍数取整": "alignment"},
    "GH_CropRestore": {"接缝": "region_info", "裁剪图像": "croped_image", "裁剪遮罩": "croped_mask", "背景图像": "background_image"},
    "图像缩放V2_孤海": {"图像": "image", "填充颜色": "fill_color", "遮罩": "mask", "宽度": "width", "高度": "height", "将边缩放到": "edge_length", "缩放方法": "size_rule", "缩放插值": "upscale_method", "缩放模式": "edit_fit", "固定方向": "crop_position", "执行条件": "execute_condition", "整除数": "divisible_by"},
    "GulfSeaImageMergeMask": {"背景图": "background_image", "覆盖图": "layer_image", "遮罩": "layer_mask", "透明度": "opacity", "遮罩扩展": "mask_expand", "模糊半径": "mask_blur", "匹配图像大小": "match_image_size"},
}
OUTPUT_SLOTS = {"GH_MaskCropV2": {0: 4, 1: 0, 2: 5}}
SCALAR_TYPES = {"INT", "FLOAT", "BOOLEAN", "STRING", "COLORCODE"}


def load_nodes(comfyui_root):
    sys.path.insert(0, str(comfyui_root))
    from comfy.cli_args import args
    args.cpu = True
    from server import PromptServer
    if not hasattr(PromptServer, "instance"):
        PromptServer(asyncio.new_event_loop())
    spec = importlib.util.spec_from_file_location("swwan_migration", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def defaults(cls):
    values = {}
    for group in ("required", "optional"):
        for name, data in cls.INPUT_TYPES().get(group, {}).items():
            kind = data[0]
            if isinstance(kind, list) or kind in SCALAR_TYPES:
                options = data[1] if len(data) > 1 else {}
                values[name] = options.get("default", kind[0] if isinstance(kind, list) else None)
    return values


def migrate(original, registry):
    result = copy.deepcopy(original)
    old_nodes = {n["id"]: n for n in original["nodes"]}
    new_nodes = {n["id"]: n for n in result["nodes"]}
    for node in result["nodes"]:
        old_type = node["type"]
        if old_type not in MAPPINGS:
            continue
        cls = registry.NODE_CLASS_MAPPINGS[MAPPINGS[old_type]]
        values = defaults(cls)
        w = node.get("widgets_values") or []
        if old_type == "GH_MaskCropV2":
            values.update(crop_mode="edit_region", fill_mask_holes=w[0],
                          top_reserve_ratio=w[1]-1, bottom_reserve_ratio=w[2]-1,
                          left_reserve_ratio=w[3]-1, right_reserve_ratio=w[4]-1,
                          output_size=w[5], custom_width=w[6], custom_height=w[7], alignment=w[8])
        elif old_type == "GH_CropRestore":
            values.update(expand_percent=w[0], feather_percent=w[1], device="CPU")
        elif old_type == "图像缩放V2_孤海":
            interpolation = {"Lanczos": "lanczos", "双线性插值": "bilinear", "双三次插值": "bicubic", "区域": "area", "邻近-精确": "nearest-exact"}
            positions = {"居中": "center", "上": "top", "下": "bottom", "左": "left", "右": "right"}
            values.update(width=w[0], height=w[1], resize_mode="edit_size", size_rule=w[2], edge_length=w[3],
                          upscale_method=interpolation[w[4]], edit_fit=w[5], crop_position=positions[w[6]],
                          execute_condition=w[7], fill_color=w[8] if len(w) > 9 else "#364254", divisible_by=w[9] if len(w) > 9 else w[8], keep_proportion="crop", device="cpu")
        elif old_type == "GulfSeaImageMergeMask":
            values.update(operation="mask_composite", opacity=int(round(w[0]*100)), invert_mask=False,
                          mask_expand=w[1], mask_blur=w[2], match_image_size=w[3], blend_mode="normal")
        else:
            for name, val in zip(values, w):
                values[name] = val
        node["type"] = MAPPINGS[old_type]
        node["title"] = registry.NODE_DISPLAY_NAME_MAPPINGS[node["type"]]
        props = node.setdefault("properties", {})
        for key in ("aux_id", "cnr_id", "ver", "ue_properties", "widget_ue_connectable"):
            props.pop(key, None)
        props["Node name for S&R"] = node["type"]
        # Persist all widgets, including hidden controls and converted inputs.
        node["widgets_values"] = list(values.values())
        node["inputs"] = []
        schema = cls.INPUT_TYPES()
        for group in ("required", "optional"):
            for name, data in schema.get(group, {}).items():
                kind = data[0]
                if isinstance(kind, list) or kind in SCALAR_TYPES:
                    continue
                entry = {"name": name, "type": kind, "link": None}
                if group == "optional": entry["shape"] = 7
                node["inputs"].append(entry)
        names = getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)
        node["outputs"] = [{"name": name, "type": kind, "links": None}
                           for name, kind in zip(names, cls.RETURN_TYPES)]
    # Rebuild connections by input names rather than trusting old positional slots.
    for link in result["links"]:
        _, source, slot, target, target_slot, kind = link
        old_src, old_target = old_nodes[source], old_nodes[target]
        if old_src["type"] in MAPPINGS:
            link[2] = OUTPUT_SLOTS.get(old_src["type"], {}).get(slot, slot)
        if old_target["type"] in MAPPINGS:
            old_name = old_target["inputs"][target_slot]["name"]
            name = INPUT_NAMES.get(old_target["type"], {}).get(old_name, old_name)
            inputs = new_nodes[target]["inputs"]
            if not any(i["name"] == name for i in inputs):
                inputs.append({"name": name, "type": kind, "link": None, "widget": {"name": name}})
            link[4] = next(i for i, entry in enumerate(inputs) if entry["name"] == name)
    for old in original["nodes"]:
        if old["type"] != "GH_CropRestore": continue
        restore = new_nodes[old["id"]]
        crop_link = next(link for link in original["links"] if link[3] == old["id"] and link[5] == "SEAM")
        crop_id = crop_link[1]
        image_link = next(link for link in original["links"] if link[3] == crop_id and link[4] == 0)
        for source, slot, name, kind in [(image_link[1], image_link[2], "background_image", "IMAGE"), (crop_id, 2, "crop_box", "BOX")]:
            index = next(i for i, entry in enumerate(restore["inputs"]) if entry["name"] == name)
            if any(link[3] == old["id"] and link[4] == index for link in result["links"]): continue
            result["last_link_id"] += 1
            result["links"].append([result["last_link_id"], source, slot, old["id"], index, kind])
    # Keep untouched core/Qwen nodes byte-for-byte except a necessary added source link.
    for node in result["nodes"]:
        if old_nodes[node["id"]]["type"] not in MAPPINGS:
            continue
        for inp in node.get("inputs", []):
            if inp.get("link") is not None: inp["link"] = None
        for out in node.get("outputs", []): out["links"] = None
    for link_id, source, slot, target, target_slot, kind in result["links"]:
        out = new_nodes[source]["outputs"][slot]
        if link_id not in (out.get("links") or []):
            out["links"] = (out.get("links") or []) + [link_id]
        new_nodes[target]["inputs"][target_slot]["link"] = link_id
    from migrate_workflow import migrate as normalize
    return normalize(result, registry=registry)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    opts = parser.parse_args()
    if opts.input.resolve() == opts.output.resolve(): parser.error("Output must differ from input")
    registry = load_nodes(opts.comfyui_root)
    migrated = migrate(json.loads(opts.input.read_text()), registry)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved migrated workflow: {opts.output}")
