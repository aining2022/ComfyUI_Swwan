// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from '/scripts/app.js';
const NODES = new Set(['SwwanImageConcatMulti','SwwanImageBatchMulti','SwwanImageAddMulti',
    'SwwanCrossFadeImagesMulti','SwwanTransitionImagesMulti']);
export function updateInputs(node) {
    const countWidget = node.widgets?.find(w => w.name === 'inputcount');
    if (!countWidget) return;
    const socket = node.inputs?.find(i => /^(image|images)_1$/.test(i.name));
    if (!socket) return;
    const prefix = socket.name.slice(0, -1);
    const requested = Math.max(2, Math.floor(Number(countWidget.value) || 2));
    const repeated = (node.inputs || []).filter(i => i.name.startsWith(prefix));
    const lastConnected = Math.max(0, ...repeated.filter(i => i.link != null).map(i => Number(i.name.slice(prefix.length))));
    const count = Math.max(requested, lastConnected);
    // Keep the executable count consistent when connected inputs prevent shrinking.
    countWidget.value = count;
    for (let i = 1; i <= count; i++) {
        if (!node.inputs.some(input => input.name === prefix + i)) node.addInput(prefix + i, socket.type);
    }
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        if (input.name.startsWith(prefix) && Number(input.name.slice(prefix.length)) > count && input.link == null) node.removeInput(i);
    }
    node.setDirtyCanvas?.(true, true);
}
app.registerExtension({
    name: 'Swwan.DynamicInputs',
    beforeRegisterNodeDef(Node, data) {
        if (!NODES.has(data.name)) return;
        for (const event of ['onNodeCreated', 'onConfigure', 'onConnectionsChange']) {
            const original = Node.prototype[event];
            Node.prototype[event] = function() {
                const result = original?.apply(this, arguments);
                if (event === 'onNodeCreated') {
                    const widget = this.widgets?.find(w => w.name === 'inputcount');
                    if (widget) {const callback = widget.callback; widget.callback = (...args) => {callback?.apply(widget, args); updateInputs(this);};}
                }
                updateInputs(this); return result;
            };
        }
    },
});
