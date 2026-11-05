// SPDX-License-Identifier: MIT
import { app } from '../../scripts/app.js';

const controlsId = 'SwwanQwen21PrunaControls';
const inputsId = 'SwwanQwen21PrunaInputs';
const value = (node, name) => node.widgets?.find(w => w.name === name)?.value;
const connected = (node, name) => node.inputs?.some(i => i.name === name && i.link != null);
function visible(widget, show) {
    if (widget.type?.startsWith('converted-widget')) return;
    widget.prunaDisplay ??= {type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden};
    widget.type = show ? widget.prunaDisplay.type : 'swwan-hidden';
    widget.hidden = show ? widget.prunaDisplay.hidden : true;
    widget.computeSize = show ? widget.prunaDisplay.computeSize : () => [0, -4];
}
function source(node) {
    const input = node.inputs?.find(i => i.name === 'config');
    const link = node.graph?.links?.[input?.link];
    const origin = link && node.graph?.getNodeById(link.origin_id);
    return origin?.comfyClass === controlsId ? origin : null;
}
function update(node) {
    const config = node.comfyClass === controlsId ? node : source(node);
    const hide = new Set();
    if (config && !connected(config, 'task')) {
        const task = value(config, 'task');
        const text = task === 'photoreal' && !connected(config, 'generation_mode') && value(config, 'generation_mode') === 'text';
        if (node.comfyClass === inputsId) {
            if (text) ['image_filename','mask_filename','protect_filename','reference_filenames'].forEach(n => hide.add(n));
            else {
                if (!['edit','face_swap'].includes(task)) hide.add('reference_filenames');
                if (!['remove','cutout','face_swap'].includes(task) && !( ['edit','watermark'].includes(task) && (connected(config,'edit_scope') || value(config,'edit_scope') === 'local'))) hide.add('mask_filename');
            }
        } else {
            const groups = {
                edit_scope: ['edit','watermark'], generation_mode: ['photoreal'],
                marking: ['edit','watermark','remove','cutout'], enhanced: ['remove'], mark_opacity: ['remove'], block_size: ['remove'],
                padding_unit: ['outpaint'], left: ['outpaint'], right: ['outpaint'], top: ['outpaint'], bottom: ['outpaint'], seam_band: ['outpaint'],
                people: ['face_swap'], swap_head: ['face_swap'], forehead: ['face_swap'], skin_match: ['face_swap'], skin_strength: ['face_swap'],
                azimuth: ['angles'], elevation: ['angles'], close_up: ['angles'], reference_resolution: ['edit','face_swap'],
            };
            for (const [name, tasks] of Object.entries(groups)) if (!tasks.includes(task)) hide.add(name);
            for (const widget of node.widgets || []) if (widget.name.startsWith('task_lora_') && widget.name !== `task_lora_${task}`) hide.add(widget.name);
            if (!text && !(task === 'photoreal' && connected(config,'generation_mode'))) ['output_width','output_height'].forEach(n => hide.add(n));
            if (text) ['qwen_min_resolution','qwen_max_resolution','qwen_tolerance','context_factor','color_mode','color_strength','expand_percent','feather_percent'].forEach(n => hide.add(n));
            if (task === 'cutout') ['color_mode','color_strength','expand_percent','feather_percent'].forEach(n => hide.add(n));
            if (task === 'face_swap') ['color_mode','color_strength','marking','context_factor'].forEach(n => hide.add(n));
            if (['edit','watermark'].includes(task) && !connected(config,'edit_scope') && value(config,'edit_scope') === 'full') hide.add('marking');
            if (task === 'face_swap' && !connected(config,'skin_match') && !value(config,'skin_match')) hide.add('skin_strength');
            if (task === 'remove' && !connected(config,'enhanced')) hide.add(value(config,'enhanced') ? 'mark_opacity' : 'block_size');
            if (!connected(config,'color_mode') && value(config,'color_mode') === 'none') hide.add('color_strength');
            if (!connected(config,'additional_lora_mode') && (value(config,'additional_lora_mode') === 'off' || value(config,'additional_lora_mode') === 'edit_only' && task !== 'edit')) ['additional_lora_name','additional_lora_strength'].forEach(n => hide.add(n));
        }
    }
    for (const widget of node.widgets || []) visible(widget, !hide.has(widget.name));
    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}
function refresh(node) {
    update(node);
    for (const output of node.outputs || []) for (const id of output.links || []) {
        const link = node.graph?.links?.[id];
        const target = link && node.graph?.getNodeById(link.target_id);
        if (target?.comfyClass === inputsId) update(target);
    }
}
app.registerExtension({
    name: 'Swwan.PrunaTaskControls',
    beforeRegisterNodeDef(Node, data) {
        if (![controlsId, inputsId].includes(data.name)) return;
        Node.prototype.swwanRefreshModes = function() { refresh(this); };
        const created = Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            for (const widget of this.widgets || []) {
                const callback = widget.callback;
                widget.callback = (...args) => { const r = callback?.apply(widget, args); refresh(this); return r; };
            }
            refresh(this);
            return result;
        };
        for (const event of ['onConfigure','onConnectionsChange']) {
            const original = Node.prototype[event];
            Node.prototype[event] = function() { const r = original?.apply(this, arguments); refresh(this); return r; };
        }
    },
});
