"""Migrate this user-supplied Qwen face/head workflow without changing model or loop logic.

Only pure image/mask processing is selected. LG_Color_Match_V2 is explicitly excluded.
The original Downloads file is never written; the command rejects input==output and existing outputs.
"""
import argparse
import copy
import json
from pathlib import Path
from migrate_qwen2511_workflow import MAPPINGS as BASE, INPUT_NAMES as BASE_NAMES, defaults, load_nodes, migrate as base_migrate

MAPPINGS = {**BASE,
    "DrawMaskOnImage": "SwwanDrawMaskOnImage",
    "Mask Fill Holes": "SwwanMaskProcess", "ToBinaryMask": "SwwanMaskProcess",
    "MaskFix+": "SwwanMaskProcess", "LayerMask: MaskGrow": "SwwanMaskProcess", "GrowMaskWithBlur": "SwwanMaskProcess",
    "孤海-遮罩混合运算": "SwwanMaskCombine", "孤海遮罩分析": "SwwanMaskAnalyze", "GuHaiMaskDetect": "SwwanMaskAnalyze",
    "MaskToSEGS": "SwwanMaskSegments", "SegsToCombinedMask": "SwwanMaskSegments", "ImpactSEGSToMaskBatch": "SwwanMaskSegments", "孤海Seg次序过滤": "SwwanMaskSegments",
    "RemoveBackgroundWithMask": "SwwanImageMatte", "ImageMaskPreview_Guhai": "SwwanImageAndMaskPreview",
    "ImageColorMatch+": "SwwanColorMatch", "ImageResize+": "ImageResizeKJv2Alternative",
}
INPUT_NAMES = {
    "Mask Fill Holes": {"masks": "mask"}, "MaskFix+": {"fill_holes": "close_holes"},
    "孤海-遮罩混合运算": {"遮罩1": "mask_1", "遮罩2": "mask_2", "混合模式": "operation", "BBOX": "bbox_mode", "对齐方式": "alignment"},
    "孤海遮罩分析": {"遮罩": "mask", "上扩百分比": "top_percent", "下扩百分比": "bottom_percent", "左扩百分比": "left_percent", "右扩百分比": "right_percent"},
    "GuHaiMaskDetect": {"遮罩": "mask", "过滤最小值": "minimum_area_percent"},
    "孤海Seg次序过滤": {"Seg": "segs", "优先规则": "sort_rule", "正反顺序": "sort_order", "开始索引": "start_index", "过滤数量": "count", "分组阈值": "group_threshold"},
    "RemoveBackgroundWithMask": {"图像": "image", "遮罩": "mask", "遮罩填充漏洞": "fill_holes", "遮罩裁剪": "crop_mask", "裁剪系数": "crop_factor", "图像描边": "stroke_width", "描边颜色": "background_color"},
    "ImageMaskPreview_Guhai": {"图像": "image", "遮罩": "mask", "遮罩不透明": "mask_opacity", "遮罩颜色": "region_color", "显示序号": "show_numbers", "序号不透明": "number_opacity", "序号缩放": "number_scale", "字号比例": "font_scale", "序号字体": "number_font", "序号颜色": "number_color"},
    "ImageColorMatch+": {"image": "image_target", "reference": "image_ref"},
    "ImageResize+": {"interpolation": "essentials_interpolation", "method": "essentials_method", "condition": "essentials_condition", "multiple_of": "divisible_by"},
}
OUTPUT_SLOTS = {"GuHaiMaskDetect": {0: 7}, "SegsToCombinedMask": {0: 1}, "ImpactSEGSToMaskBatch": {0: 1}}


def replace_node(node, cls, registry, values):
    old = copy.deepcopy(node)
    node["type"] = MAPPINGS[old["type"]]
    node["title"] = registry.NODE_DISPLAY_NAME_MAPPINGS[node["type"]]
    props = node.setdefault("properties", {})
    for key in ("aux_id", "cnr_id", "ver", "ue_properties", "widget_ue_connectable"): props.pop(key, None)
    props["Node name for S&R"] = node["type"]
    props["swwan_version"] = registry.__version__
    props["cnr_id"] = "comfyui_swwan"
    node["widgets_values"] = list(values.values())
    node["inputs"] = []
    for group in ("required", "optional"):
        for name, data in cls.INPUT_TYPES().get(group, {}).items():
            if isinstance(data[0], list) or data[0] in {"INT", "FLOAT", "BOOLEAN", "STRING", "COLORCODE"}: continue
            inp = {"name": name, "type": data[0], "link": None}
            if group == "optional": inp["shape"] = 7
            node["inputs"].append(inp)
    node["outputs"] = [{"name": name, "type": kind, "links": None} for name, kind in zip(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES), cls.RETURN_TYPES)]


def migrate(original, registry):
    # base migration is already idempotent and remaps crop SEAM, native MASK and original-image/BOX restore inputs.
    result = base_migrate(original, registry)
    nodes = {n["id"]: n for n in result["nodes"]}
    old_nodes = {n["id"]: n for n in original["nodes"]}
    selected = set()
    for old in original["nodes"]:
        kind = old["type"]
        if kind not in MAPPINGS or kind in BASE: continue
        node = nodes[old["id"]]
        cls = registry.NODE_CLASS_MAPPINGS[MAPPINGS[kind]]
        v = defaults(cls); w = old.get("widgets_values") or []
        if kind == "Mask Fill Holes": v.update(operation="fill_holes")
        elif kind == "ToBinaryMask": v.update(operation="binary", threshold=w[0])
        elif kind == "MaskFix+": v.update(operation="cleanup", erode_dilate=w[0], close_holes=w[1], remove_isolated_pixels=w[2], smooth=w[3], blur=w[4])
        elif kind == "LayerMask: MaskGrow": v.update(operation="layer_grow", invert_mask=w[0], grow=w[1], blur=w[2])
        elif kind == "GrowMaskWithBlur": v.update(operation="grow_blur", **dict(zip(["expand","incremental_expandrate","tapered_corners","flip_input","blur_radius","lerp_alpha","decay_factor","fill_holes"],w)))
        elif kind == "孤海-遮罩混合运算": v.update(operation=w[0], bbox_mode=w[1], alignment=w[2])
        elif kind == "孤海遮罩分析": v.update(dict(zip(["top_percent","bottom_percent","left_percent","right_percent"],w)))
        elif kind == "GuHaiMaskDetect": v.update(minimum_area_percent=w[0])
        elif kind == "MaskToSEGS": v.update(operation="from_mask", **dict(zip(["combined","crop_factor","bbox_fill","drop_size","contour_fill"],w)))
        elif kind == "SegsToCombinedMask": v.update(operation="combined_mask")
        elif kind == "ImpactSEGSToMaskBatch": v.update(operation="mask_batch")
        elif kind == "孤海Seg次序过滤": v.update(operation="filter", **dict(zip(["sort_rule","sort_order","start_index","count","group_threshold"],w)))
        elif kind == "RemoveBackgroundWithMask": v.update(dict(zip(["fill_holes","crop_mask","crop_factor","stroke_width","background_color"],w)))
        elif kind == "ImageMaskPreview_Guhai":
            v.update(preview_mode="regions", pass_through=True, **dict(zip(["mask_opacity","region_color","show_numbers","number_opacity","number_scale","font_scale","number_font","number_color"],w)))
            v["number_font"] = "FreeMono.ttf"  # Unlicensed/missing historical font is not redistributed.
        elif kind == "ImageColorMatch+": v.update(match_mode="mean_std", **dict(zip(["color_space","factor","device","batch_size"],w)))
        elif kind == "ImageResize+": v.update(resize_mode="essentials", **dict(zip(["width","height","essentials_interpolation","essentials_method","essentials_condition","divisible_by"],w)))
        else: v.update(dict(zip(v,w)))
        replace_node(node, cls, registry, v); selected.add(old["id"])
    for link in result["links"]:
        _, source, slot, target, index, kind = link
        src, dst = old_nodes[source], old_nodes[target]
        if source in selected: link[2] = OUTPUT_SLOTS.get(src["type"], {}).get(slot, slot)
        if target in selected:
            old_name = dst["inputs"][index]["name"]
            name = INPUT_NAMES.get(dst["type"], {}).get(old_name, old_name)
            inputs = nodes[target]["inputs"]
            if not any(inp["name"] == name for inp in inputs):
                inputs.append({"name": name, "type": kind, "link": None, "widget": {"name": name}})
            link[4] = next(i for i, inp in enumerate(inputs) if inp["name"] == name)
    # Original linked crop coefficients are factors, not the existing reserve ratios.
    for old in original["nodes"]:
        if old["type"] != "GH_MaskCropV2": continue
        node = nodes[old["id"]];v = dict(zip(defaults(registry.NODE_CLASS_MAPPINGS[node["type"]]),node["widgets_values"]))
        v.update(edit_expansion_mode="factor", **dict(zip(["edit_top_factor","edit_bottom_factor","edit_left_factor","edit_right_factor"],old["widgets_values"][1:5])))
        node["widgets_values"] = list(v.values())
        for inp in node["inputs"]:
            inp["name"] = {"top_reserve_ratio":"edit_top_factor","bottom_reserve_ratio":"edit_bottom_factor","left_reserve_ratio":"edit_left_factor","right_reserve_ratio":"edit_right_factor"}.get(inp["name"],inp["name"])
            if "widget" in inp: inp["widget"]["name"] = inp["name"]
    # Rebuild only migrated connection records; untouched node parameters remain identical.
    migrated_ids = {n["id"] for n in original["nodes"] if n["type"] in MAPPINGS}
    for node_id in migrated_ids:
        for inp in nodes[node_id]["inputs"]: inp["link"] = None
        for out in nodes[node_id]["outputs"]: out["links"] = None
    for link_id, source, slot, target, index, kind in result["links"]:
        out = nodes[source]["outputs"][slot]
        if link_id not in (out.get("links") or []): out["links"] = (out.get("links") or []) + [link_id]
        nodes[target]["inputs"][index]["link"] = link_id
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("output",type=Path);p.add_argument("--comfyui-root",type=Path,required=True)
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve() or args.output.exists(): p.error("Choose a new output path; source and existing files are never overwritten")
    reg=load_nodes(args.comfyui_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(migrate(json.loads(args.input.read_text()),reg),ensure_ascii=False,indent=2)+'\n')
    print(f"Saved {args.output}")
