// Evaluate on an isolated ComfyUI CPU service; no model nodes or inference.
(async () => {
    const {app} = await import('./scripts/app.js');
    const check = (ok, message) => { if (!ok) throw Error(message); };
    const widget = (node, name) => node.widgets.find(w => w.name === name);
    const set = (node, name, value) => {
        const w = widget(node, name); check(w, `Missing ${name}`);
        w.value = value; w.callback?.(value);
    };
    const visible = (node, name) => !widget(node, name).hidden && widget(node, name).type !== 'swwan-hidden';
    const make = type => { const node = LiteGraph.createNode(type); check(node, type); app.graph.add(node); return node; };
    const connect = (source, slot, target, name) => {
        const input = target.inputs.findIndex(i => i.name === name);
        check(input >= 0 && source.connect(slot, target, input), `Connect ${name}`);
    };
    app.graph.clear();
    const crop = make('SwwanCropByMaskV5');
    check(widget(crop, 'qwen_min_resolution').value === 128, 'Default floor');
    check(widget(crop, 'qwen_max_resolution').value === 1024, 'Default cap');
    const response = await fetch('./object_info/SwwanCropByMaskV5');
    const info = (await response.json()).SwwanCropByMaskV5;
    check(info.description.includes('qwen_min_resolution=128') && info.description.includes('qwen_max_resolution=1024'), 'Updated node description');
    check(info.output_tooltips.length === 12, 'Output help');
    for (const group of Object.values(info.input)) {
        for (const [name, data] of Object.entries(group)) check(data[1]?.tooltip?.length > 20, `Input help ${name}`);
    }
    check(crop.constructor.nodeData.description === info.description, 'Frontend received description');
    check(crop.constructor.nodeData.input.optional.qwen_min_resolution[1].tooltip.includes('默认 128'), 'Frontend received minimum tooltip');
    set(crop, 'crop_mode', 'qwen21_aligned');
    set(crop, 'qwen_resolution_mode', 'crop_area');
    set(crop, 'qwen_min_resolution', 384);
    set(crop, 'qwen_max_resolution', 1560);
    check(visible(crop, 'qwen_min_resolution'), 'Dynamic floor visible');
    set(crop, 'qwen_resolution_mode', 'fixed');
    check(!visible(crop, 'qwen_min_resolution'), 'Fixed floor hidden');
    set(crop, 'qwen_resolution_mode', 'crop_area');
    check(widget(crop, 'qwen_min_resolution').value === 384, 'Hidden value preserved');
    const image = make('LayerUtility: ColorImage (Swwan)');
    set(image, 'width', 101); set(image, 'height', 73);
    const mask = make('SolidMask');
    set(mask, 'width', 101); set(mask, 'height', 73); set(mask, 'value', 1);
    connect(image, 0, crop, 'image'); connect(mask, 0, crop, 'mask');
    const preview = make('PreviewImage'); connect(crop, 0, preview, 'images');
    const id = crop.id;
    const first = await app.graphToPrompt();
    check(first.output[id].inputs.qwen_min_resolution === 384, 'Literal API floor');
    const saved = JSON.parse(JSON.stringify(app.graph.serialize()));
    await app.loadGraphData(saved);
    check(JSON.stringify((await app.graphToPrompt()).output) === JSON.stringify(first.output), 'Literal reload');
    const restored = app.graph.getNodeById(id);
    const floor = make('MathExpression_UTK'); set(floor, 'expression', '257');
    connect(floor, 0, restored, 'qwen_min_resolution');
    const linked = await app.graphToPrompt();
    check(Array.isArray(linked.output[id].inputs.qwen_min_resolution), 'Linked API floor');
    const linkedSaved = JSON.parse(JSON.stringify(app.graph.serialize()));
    await app.loadGraphData(linkedSaved);
    check(JSON.stringify((await app.graphToPrompt()).output) === JSON.stringify(linked.output), 'Linked reload');
    const legacy = JSON.parse(JSON.stringify(saved));
    const old = legacy.nodes.find(n => String(n.id) === String(id));
    const index = crop.widgets.findIndex(w => w.name === 'qwen_min_resolution');
    check(index === old.widgets_values.length - 1, 'Appended floor preserves old positions');
    old.widgets_values.pop();
    if (old.widgets_values_named) delete old.widgets_values_named.qwen_min_resolution;
    old.inputs = old.inputs.filter(i => i.name !== 'qwen_min_resolution');
    await app.loadGraphData(legacy);
    check(widget(app.graph.getNodeById(id), 'qwen_min_resolution').value === 128, 'Missing field uses new default');
    check((await app.graphToPrompt()).output[id].inputs.qwen_max_resolution === 1560, 'Old maximum retained');
    window.qwenBudgetAcceptance = {status: 'PASS', checks: [
        'defaults128/1024', 'Chinese input/output help and frontend description', 'dynamic/fixed visibility', 'hidden values preserved',
        'literal API and save/reload', 'connected floor and save/reload',
        'old workflow missing appended input defaults to128', 'old maximum retained',
    ]};
    return window.qwenBudgetAcceptance;
})()
