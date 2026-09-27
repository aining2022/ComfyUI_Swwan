// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';
const MAX = 1125899906842624;
function randomSeed() { return 1 + Math.floor(Math.random() * MAX); }
export function resolveSeed(value, last, random = randomSeed) {
    if (![ -1, -2, -3 ].includes(value)) return value;
    if (value === -1 || !Number.isSafeInteger(last) || last < 0) return random();
    return value === -2 ? (last + 1) % (MAX + 1) : (last + MAX) % (MAX + 1);
}
app.registerExtension({
    name: 'Swwan.Seed',
    setup() {
        const original = api.queuePrompt;
        api.queuePrompt = async function(number, prompt) {
            const copy = structuredClone(prompt);
            const changed = [];
            for (const [id, entry] of Object.entries(copy.output || {})) {
                if (entry.class_type !== 'SwwanSeed') continue;
                const node = app.graph?.getNodeById(Number(id));
                const value = entry.inputs.seed;
                if (typeof value !== 'number') continue;
                const actual = resolveSeed(value, node?.properties?.swwan_last_seed);
                entry.inputs.seed = actual;
                const saved = copy.workflow?.nodes?.find(n => String(n.id) === id);
                if (saved?.widgets_values) saved.widgets_values[0] = actual;
                if (saved?.widgets_values_named) saved.widgets_values_named.seed = actual;
                if (saved) (saved.properties ||= {}).swwan_last_seed = actual;
                changed.push([node, actual]);
            }
            const result = await original.call(this, number, copy);
            for (const [node, actual] of changed) {
                if (node) { (node.properties ||= {}).swwan_last_seed = actual; node.setDirtyCanvas?.(true); }
            }
            return result;
        };
    },
    beforeRegisterNodeDef(Node, data) {
        if (data.name !== 'SwwanSeed') return;
        const created = Node.prototype.onNodeCreated;
        Node.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            // ComfyUI auto-adds this stock control for a seed INT. It would
            // mutate our special mode or fixed value after the prompt is queued.
            this.widgets = this.widgets.filter(w => w.name !== 'control_after_generate');
            const seed = this.widgets.find(w => w.name === 'seed');
            for (const [name, value] of [['Random each time', -1], ['Increment each time', -2], ['Decrement each time', -3]]) {
                this.addWidget('button', name, null, () => { seed.value = value; this.setDirtyCanvas?.(true); }, {serialize:false});
            }
            this.addWidget('button', 'New fixed random', null, () => {seed.value = randomSeed(); this.setDirtyCanvas?.(true);}, {serialize:false});
            this.addWidget('button', 'Use last seed', null, () => {
                if (Number.isSafeInteger(this.properties?.swwan_last_seed)) seed.value = this.properties.swwan_last_seed;
                this.setDirtyCanvas?.(true);
            }, {serialize:false});
            return result;
        };
    },
});
