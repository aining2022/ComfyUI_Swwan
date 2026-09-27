// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from '../../scripts/app.js';

const contracts = new Map();
const scalarTypes = new Set(['INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COLORCODE']);
function validate(node, name, value, data) {
    const [kind, options = {}] = data;
    let valid = true;
    if (Array.isArray(kind)) valid = kind.includes(value);
    else if (kind === 'INT' || kind === 'FLOAT') {
        valid = typeof value === 'number' && Number.isFinite(value)
            && (kind !== 'INT' || Number.isInteger(value))
            && value >= (options.min ?? -Infinity) && value <= (options.max ?? Infinity);
    } else if (kind === 'BOOLEAN') valid = typeof value === 'boolean';
    else valid = typeof value === 'string';
    if (kind === 'COLORCODE') valid = valid && /^#(?:[\da-f]{3}|[\da-f]{6}|[\da-f]{8})$/i.test(value);
    if (!valid) throw new Error(`Swwan #${node.id}: invalid ${name}=${JSON.stringify(value)}. Repair the saved workflow parameters.`);
}
function savedValues(node, info, fields) {
    const saved = info?.widgets_values_named;
    const context = { id: info?.id ?? node.id };
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return;
    for (const name of Object.keys(saved)) {
        if (!fields.has(name)) throw new Error(`Swwan #${context.id}: unknown saved field ${name}`);
    }
    try {
        for (const [name, value] of Object.entries(saved)) validate(context, name, value, fields.get(name));
        return saved;
    } catch (error) {
        const entries = [...fields];
        const positions = info.widgets_values;
        // Recover only the exact historical omission of COLORCODE names while
        // zipping a complete canonical positional list. Truncated lists cannot
        // prove the lost final value; unrelated bad maps remain explicit errors.
        const withoutColor = entries.filter(([, spec]) => spec[0] !== 'COLORCODE');
        if (withoutColor.length === entries.length || !Array.isArray(positions)
            || positions.length !== entries.length || Object.keys(saved).length !== withoutColor.length
            || !withoutColor.every(([name], i) => Object.hasOwn(saved, name) && Object.is(saved[name], positions[i]))) throw error;
        try {
            entries.forEach(([name, spec], i) => validate(context, name, positions[i], spec));
        } catch { throw error; }
        return Object.fromEntries(entries.map(([name], i) => [name, positions[i]]));
    }
}
function collect(node, fields) {
    const values = { ...node.swwanWidgetValues };
    for (const widget of node.widgets || []) {
        if (!fields.has(widget.name)) continue;
        // Converted inputs keep their underlying value even when native
        // positional serialization omits it. UI-only widgets are not in fields.
        validate(node, widget.name, widget.value, fields.get(widget.name));
        values[widget.name] = widget.value;
    }
    for (const [name, data] of fields) {
        // Only COLORCODE needs a fallback when its custom widget is unavailable.
        // Do not synthesize parameters for arbitrary forceInput sockets.
        if (data[0] === 'COLORCODE' && !Object.hasOwn(values, name)) values[name] = data[1]?.default ?? '#364254';
    }
    node.swwanWidgetValues = values;
    return values;
}
app.registerExtension({
    name: 'Swwan.NamedWidgetValues',
    beforeRegisterNodeDef(Node, data) {
        if (!data.category?.startsWith('Swwan/')) return;
        const fields = new Map(Object.entries({ ...data.input?.required, ...data.input?.optional })
            .filter(([, spec]) => Array.isArray(spec[0]) || scalarTypes.has(spec[0])));
        contracts.set(data.name, fields);
        // Inspect before ComfyUI's configure migrates positional widget data.
        const configureNode = Node.prototype.configure;
        Node.prototype.configure = function(info) {
            const saved = savedValues(this, info, fields);
            if (saved) info.widgets_values_named = { ...saved };
            return configureNode?.apply(this, arguments);
        };
        const configure = Node.prototype.onConfigure;
        Node.prototype.onConfigure = function(info) {
            const saved = savedValues(this, info, fields);
            const result = configure?.apply(this, arguments);
            if (saved) {
                this.swwanWidgetValues = { ...saved };
                for (const widget of this.widgets || []) {
                    if (Object.hasOwn(saved, widget.name)) widget.value = saved[widget.name];
                }
            }
            // Mode visibility must not depend on extension registration order.
            this.swwanRefreshModes?.();
            return result;
        };
        const serialize = Node.prototype.onSerialize;
        Node.prototype.onSerialize = function(info) {
            const result = serialize?.apply(this, arguments);
            info.widgets_values_named = collect(this, fields);
            // Leave LiteGraph's positional list intact: it may contain native
            // frontend controls and placeholders for converted input widgets.
            return result;
        };
    },
    setup() {
        const original = app.graphToPrompt;
        app.graphToPrompt = async function() {
            const result = await original.apply(this, arguments);
            for (const [id, entry] of Object.entries(result.output || {})) {
                const fields = contracts.get(entry.class_type);
                const node = this.graph?.getNodeById(Number(id));
                if (!fields || !node) continue;
                const values = collect(node, fields);
                for (const [name, data] of fields) {
                    if (data[0] !== 'COLORCODE') continue;
                    // A real graph connection always overrides the saved literal.
                    if (node.inputs?.some(input => input.name === name && input.link != null)
                        || Array.isArray(entry.inputs[name])) continue;
                    validate(node, name, values[name], data);
                    entry.inputs[name] = values[name];
                }
            }
            return result;
        };
    },
});
