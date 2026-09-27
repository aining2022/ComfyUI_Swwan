// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from "../../scripts/app.js";

const nodeIds = new Set([
    "SwwanCropByMaskV5", "SwwanRestoreCropBoxV4",
    "ImageResizeKJv2Alternative", "ImageBlendSwwan",
    "SwwanImageConcatMulti", "SwwanSaveImage", "MathExpression_UTK",
    "SwwanMaskProcess", "SwwanMaskSegments", "SwwanImageAndMaskPreview", "SwwanColorMatch",
]);

function value(node, name) {
    return node.widgets?.find((widget) => widget.name === name)?.value;
}

function connected(node, name) {
    return node.inputs?.some((input) => input.name === name && input.link != null);
}

function setVisible(widget, visible) {
    // Converted widgets already have their own hiding and serialization logic.
    if (widget.type?.startsWith("converted-widget")) return;
    if (!widget.swwanDisplay) {
        widget.swwanDisplay = { type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden };
    }
    // Current LiteGraph uses hidden for layout and hit testing; type/size also
    // support older frontends. Neither path disables value serialization.
    widget.hidden = visible ? widget.swwanDisplay.hidden : true;
    widget.type = visible ? widget.swwanDisplay.type : "swwan-hidden";
    widget.computeSize = visible ? widget.swwanDisplay.computeSize : () => [0, -4];
}

function update(node) {
    const id = node.comfyClass || node.type;
    const hidden = new Set();
    const hide = (...names) => names.forEach((name) => hidden.add(name));
    if (id === "SwwanCropByMaskV5" && !connected(node, "crop_mode")) {
        const edit = value(node, "crop_mode") === "edit_region";
        if (edit) {
            hide("detect", "reserve_mode", "top_reserve", "bottom_reserve", "left_reserve", "right_reserve",
                "reserve_max", "round_to_multiple", "batch_mode", "device");
            if (!connected(node, "edit_expansion_mode")) {
                if (value(node, "edit_expansion_mode") === "factor") hide("top_reserve_ratio", "bottom_reserve_ratio", "left_reserve_ratio", "right_reserve_ratio");
                else hide("edit_top_factor", "edit_bottom_factor", "edit_left_factor", "edit_right_factor");
            }
            if (value(node, "output_size") !== "自定义宽高" && !connected(node, "output_size")) {
                hide("custom_width", "custom_height");
            }
        } else {
            hide("fill_mask_holes", "output_size", "custom_width", "custom_height", "alignment", "edit_expansion_mode", "edit_top_factor", "edit_bottom_factor", "edit_left_factor", "edit_right_factor");
            if (!connected(node, "reserve_mode") && value(node, "reserve_mode") === "absolute") {
                hide("top_reserve_ratio", "bottom_reserve_ratio", "left_reserve_ratio", "right_reserve_ratio", "reserve_max");
            }
        }
    } else if (id === "ImageResizeKJv2Alternative" && !connected(node, "resize_mode")) {
        const aspectNames = ["aspect_ratio", "proportional_width", "proportional_height", "aspect_fit", "aspect_method", "aspect_round", "aspect_scale_side", "aspect_length"];
        const essentials = ["essentials_method", "essentials_condition", "essentials_interpolation"];
        if (value(node, "resize_mode") !== "essentials") hide(...essentials);
        if (value(node, "resize_mode") === "essentials") {
            hide("upscale_method", "keep_proportion", "pad_color", "crop_position", "device", "size_rule", "edge_length", "execute_condition", "edit_fit", "fill_color", ...aspectNames);
        } else if (value(node, "resize_mode") === "aspect_ratio") {
            hide("width", "height", "upscale_method", "keep_proportion", "pad_color", "crop_position", "divisible_by", "device", "size_rule", "edge_length", "execute_condition", "edit_fit");
            if (value(node, "aspect_ratio") !== "custom" && !connected(node, "aspect_ratio")) hide("proportional_width", "proportional_height");
        } else if (value(node, "resize_mode") === "edit_size") {
            hide(...aspectNames);
            hide("keep_proportion", "pad_color", "device");
            if (!connected(node, "size_rule")) {
                if (value(node, "size_rule") === "自定义宽高") {
                    if (!connected(node, "execute_condition") && value(node, "execute_condition") === "总是") hide("edge_length");
                }
                else hide("width", "height");
            }
        } else {
            hide("size_rule", "edge_length", "execute_condition", "edit_fit", "fill_color", ...aspectNames);
        }
    } else if (id === "SwwanMaskProcess" && !connected(node, "operation")) {
        const fields = {
            binary: ["threshold"], fill_holes: [],
            cleanup: ["erode_dilate", "close_holes", "remove_isolated_pixels", "smooth", "blur"],
            layer_grow: ["invert_mask", "grow", "blur"],
            grow_blur: ["expand", "incremental_expandrate", "tapered_corners", "flip_input", "blur_radius", "lerp_alpha", "decay_factor", "fill_holes"],
        };
        const active = new Set(fields[value(node, "operation")] || []);
        for (const name of new Set(Object.values(fields).flat())) if (!active.has(name)) hide(name);
    } else if (id === "SwwanMaskSegments" && !connected(node, "operation")) {
        const convert = ["combined", "crop_factor", "bbox_fill", "drop_size", "contour_fill"];
        const filter = ["sort_rule", "sort_order", "start_index", "count", "group_threshold"];
        if (value(node, "operation") !== "from_mask") hide(...convert);
        if (value(node, "operation") !== "filter") hide(...filter);
    } else if (id === "SwwanImageAndMaskPreview" && !connected(node, "preview_mode")) {
        if (value(node, "preview_mode") === "regions") hide("mask_color", "pass_through");
        else hide("region_color", "show_numbers", "number_opacity", "number_scale", "font_scale", "number_font", "number_color");
    } else if (id === "SwwanColorMatch" && !connected(node, "match_mode")) {
        if (value(node, "match_mode") === "mean_std") hide("method", "strength", "multithread");
        else hide("color_space", "factor", "device", "batch_size");
    } else if (id === "ImageBlendSwwan" && !connected(node, "operation")) {
        if (value(node, "operation") === "mask_composite") hide("blend_mode");
        else hide("mask_expand", "mask_blur", "match_image_size");
    } else if (id === "SwwanImageConcatMulti" && !connected(node, "layout")) {
        const layout = value(node, "layout");
        if (layout === "strip") hide("columns");
        else hide("direction");
        if (layout === "batch_grid") hide("inputcount", "match_image_size");
    } else if (id === "MathExpression_UTK" && !connected(node, "preset")) {
        if (value(node, "preset") !== "custom") hide("expression");
    } else if (id === "SwwanSaveImage" && !connected(node, "file_format")) {
        const format = value(node, "file_format");
        if (format !== "png") hide("png_compress_level");
        if (format !== "webp") hide("webp_lossless", "webp_method");
        if (format !== "webp" && format !== "jpg") hide("quality");
    } else if (id === "SwwanRestoreCropBoxV4") {
        const input = node.inputs?.find((input) => input.name === "region_info");
        const link = node.graph?.links?.[input?.link];
        const source = link && node.graph?.getNodeById(link.origin_id);
        const known = source && (source.comfyClass || source.type) === "SwwanCropByMaskV5"
            && !connected(source, "crop_mode");
        const edit = known && value(source, "crop_mode") === "edit_region";
        if (edit) hide("feathering", "device");
        if (!connected(node, "region_info") || (known && !edit)) hide("expand_percent", "feather_percent");
    }
    for (const widget of node.widgets || []) setVisible(widget, !hidden.has(widget.name));
    // Keep the user's width and let the node compute its needed height.
    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function refresh(node) {
    update(node);
    for (const output of node.outputs || []) {
        for (const linkId of output.links || []) {
            const link = node.graph?.links?.[linkId];
            const target = link && node.graph?.getNodeById(link.target_id);
            if (target && nodeIds.has(target.comfyClass || target.type)) update(target);
        }
    }
}

app.registerExtension({
    name: "Swwan.WorkflowImageModes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeIds.has(nodeData.name)) return;
        nodeType.prototype.swwanRefreshModes = function () { refresh(this); };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            for (const widget of this.widgets || []) {
                const callback = widget.callback;
                widget.callback = (...args) => {
                    const result = callback?.apply(widget, args);
                    refresh(this);
                    return result;
                };
            }
            refresh(this);
            return result;
        };
        for (const event of ["onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[event];
            nodeType.prototype[event] = function () {
                const result = original?.apply(this, arguments);
                refresh(this);
                return result;
            };
        }
    },
});
