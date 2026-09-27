import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../', import.meta.url));
const extensions = [];
const elements = [];
const app = { registerExtension: (extension) => extensions.push(extension) };
const context = vm.createContext({
    app,
    ComfyWidgets: {
        STRING(node, name, data) {
            const widget = { name, type: 'text', value: data[1].default, options: {}, callback() {} };
            node.widgets.push(widget);
            return { widget };
        },
    },
    document: {
        addEventListener() {}, removeEventListener() {},
        body: { appendChild: (element) => elements.push(element) },
        createElement() {
            return {
                style: {}, listeners: {}, removed: false,
                addEventListener(name, callback) { this.listeners[name] = callback; },
                click() { this.clicked = true; },
                remove() { this.removed = true; },
            };
        },
    },
});
for (const name of ['workflow_image_modes', 'swwan_color_widget']) {
    const source = readFileSync(`${root}web/js/${name}.js`, 'utf8').replace(/^import .*;\n/gm, '');
    vm.runInContext(source, context, { filename: name });
}
const modes = extensions.find((extension) => extension.name === 'Swwan.WorkflowImageModes');
const colors = extensions.find((extension) => extension.name === 'Swwan.ColorCodeWidget');
function makeNode(id, values) {
    class Node {
        constructor() {
            this.comfyClass = id;
            this.widgets = Object.entries(values).map(([name, value]) => ({ name, value, type: 'combo', options: {} }));
            this.inputs = [];
            this.outputs = [];
            this.size = [320, 200];
        }
        computeSize() { return [300, this.widgets.filter((widget) => widget.type !== 'swwan-hidden').length * 20 + 60]; }
        setSize(size) { this.size = size; }
        setDirtyCanvas() {}
    }
    modes.beforeRegisterNodeDef(Node, { name: id });
    const node = new Node();
    node.onNodeCreated();
    return node;
}
const widget = (node, name) => node.widgets.find((widget) => widget.name === name);
const visible = (node, name) => !widget(node, name).hidden && widget(node, name).type !== 'swwan-hidden';
const set = (node, name, value) => { widget(node, name).value = value; widget(node, name).callback(value); };
const savedValues = (node) => JSON.stringify(node.widgets.map((widget) => widget.value));

const crop = makeNode('SwwanCropByMaskV5', {
    crop_mode: 'bounds', detect: 'mask_area', reserve_mode: 'absolute', reserve_max: 100,
    top_reserve: 10, bottom_reserve: 10, left_reserve: 10, right_reserve: 10,
    top_reserve_ratio: 0.3, bottom_reserve_ratio: 0.3, left_reserve_ratio: 0.3, right_reserve_ratio: 0.3,
    fill_mask_holes: true, output_size: '原像素', custom_width: 1024, custom_height: 768,
    alignment: 8, device: 'cpu', round_to_multiple: '8', batch_mode: 'single_frame',
});
assert.equal(visible(crop, 'custom_width'), false);
assert.equal(visible(crop, 'top_reserve_ratio'), false);
const restore = makeNode('SwwanRestoreCropBoxV4', { feathering: 8, device: 'cpu', expand_percent: 5, feather_percent: 5 });
assert.equal(visible(restore, 'expand_percent'), false);
crop.outputs = [{ links: [1] }];
restore.inputs = [{ name: 'region_info', link: 1 }];
const graph = { links: { 1: { origin_id: 1, target_id: 2 } }, getNodeById: (id) => id === 1 ? crop : restore };
crop.graph = restore.graph = graph;
set(crop, 'crop_mode', 'edit_region');
assert.equal(visible(crop, 'top_reserve_ratio'), true);
assert.equal(visible(crop, 'detect'), false);
assert.equal(visible(restore, 'expand_percent'), true);
assert.equal(visible(restore, 'feathering'), false);
set(crop, 'output_size', '自定义宽高');
assert.equal(visible(crop, 'custom_width'), true);
const editSaved = savedValues(crop);
// Loading a saved graph uses the same complete widget order even when hidden.
crop.onConfigure();
assert.equal(savedValues(crop), editSaved);
set(crop, 'crop_mode', 'bounds');
assert.equal(visible(restore, 'feathering'), true);
assert.equal(widget(crop, 'custom_height').value, 768);
const beforeSave = savedValues(crop);
crop.onConfigure();
assert.equal(savedValues(crop), beforeSave);
// A connected selector has runtime values, so keep both sets available.
crop.inputs.push({ name: 'crop_mode', link: 3 });
crop.onConnectionsChange();
assert.equal(visible(crop, 'fill_mask_holes'), true);
assert.equal(visible(crop, 'detect'), true);

const resize = makeNode('ImageResizeKJv2Alternative', {
    resize_mode: 'standard', size_rule: '按长边等比例', width: 640, height: 480,
    keep_proportion: 'pad', pad_color: '0, 0, 0', device: 'cpu', edge_length: 1024,
    execute_condition: '总是', edit_fit: '裁剪', fill_color: '#364254',
});
assert.equal(visible(resize, 'edge_length'), false);
set(resize, 'resize_mode', 'edit_size');
assert.equal(visible(resize, 'width'), false);
set(resize, 'size_rule', '自定义宽高');
assert.equal(visible(resize, 'width'), true);
assert.equal(visible(resize, 'edge_length'), false);
set(resize, 'execute_condition', '最长边大于时');
assert.equal(visible(resize, 'edge_length'), true);
const converted = widget(resize, 'fill_color');
converted.type = 'converted-widget:text';
set(resize, 'resize_mode', 'standard');
assert.equal(converted.type, 'converted-widget:text');
assert.equal(converted.value, '#364254');
const blend = makeNode('ImageBlendSwwan', { operation: 'blend', blend_mode: 'normal', mask_expand: 5, mask_blur: 3, match_image_size: true });
assert.equal(visible(blend, 'mask_expand'), false);
set(blend, 'operation', 'mask_composite');
assert.equal(visible(blend, 'blend_mode'), false);
assert.equal(visible(blend, 'mask_expand'), true);

const grid = makeNode('SwwanImageConcatMulti',{layout:'strip',columns:2,direction:'right',inputcount:4,match_image_size:false});
assert.equal(visible(grid,'columns'),false);set(grid,'layout','grid');assert.equal(visible(grid,'columns'),true);assert.equal(visible(grid,'direction'),false);
const math = makeNode('MathExpression_UTK',{preset:'custom',expression:'a+b'});set(math,'preset','a+b');assert.equal(visible(math,'expression'),false);assert.equal(widget(math,'expression').value,'a+b');
const save = makeNode('SwwanSaveImage',{file_format:'png',quality:88,png_compress_level:1,webp_lossless:false,webp_method:2});
assert.equal(visible(save,'quality'),false);set(save,'file_format','webp');assert.equal(visible(save,'quality'),true);assert.equal(visible(save,'png_compress_level'),false);

// Both extension orders use valid COLORCODE widgets and keep the typed socket.
const originalPath = process.argv[2];
let original;
if (originalPath) {
    vm.runInContext(readFileSync(originalPath, 'utf8').replace(/^import .*;\n/gm, ''), context);
    original = extensions.find((extension) => extension.name === 'AILab.colorWidget');
}
for (const order of original ? [[original, colors], [colors, original]] : [[colors]]) {
    const registry = Object.assign({}, ...order.map((extension) => extension.getCustomWidgets()));
    const node = { widgets: [], graph: { _version: 1 }, size: [320, 200], setDirtyCanvas() {}, addCustomWidget(widget) { this.widgets.push(widget); return widget; } };
    const result = registry.COLORCODE(node, 'fill_color', ['COLORCODE', { default: '#123456' }]);
    assert.equal(result.widget.value, '#123456');
    assert.equal(result.widget.options.widgetType || result.widget.type, 'COLORCODE');
    assert.deepEqual(JSON.parse(savedValues(node)), ['#123456']);
    result.widget.mouse({ type: 'pointerdown' }, [50, 10], node);
    const picker = elements.at(-1);
    assert.equal(picker.type, 'color');
    assert.equal(picker.clicked, true);
    picker.value = '#abcdef';
    picker.listeners.input?.();
    picker.listeners.change();
    assert.equal(result.widget.value, '#abcdef');
    assert.equal(picker.removed, true);
    assert.deepEqual(JSON.parse(savedValues(node)), ['#abcdef']);
}
console.log('PASS: modes, connected inputs, restore visibility, serialized values and COLORCODE widgets' + (original ? ' (both plugin orders)' : ''));
