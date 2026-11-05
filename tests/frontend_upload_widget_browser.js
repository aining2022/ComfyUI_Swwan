// SPDX-License-Identifier: MIT
// Run only on an isolated ComfyUI page. No model or GPU inference is queued.
(async (source) => {
    const {app} = await import(new URL('scripts/app.js', document.baseURI).href);
    const vue = [...document.querySelectorAll('div')].find(el => el.__vue_app__)?.__vue_app__;
    const context = fn => vue ? vue.runWithContext(fn) : fn();
    const checks = [];
    const check = (ok, label) => {if (!ok) throw Error(label); checks.push(label);};
    const nodeInfo = await (await fetch('/object_info/SwwanLoadAndResizeImage')).json();
    check(!nodeInfo.SwwanLoadAndResizeImage.input.required.image[0].includes('selection.png'),
          'selection.png really absent from server inventory');
    // Use the exact failing v5 node and its outgoing image->mask connection,
    // without pretending the isolated server has all Qwen/BFS dependencies.
    const loader = structuredClone(source.nodes.find(n => n.type === 'SwwanLoadAndResizeImage'));
    const edge = source.links.find(l => l[1] === loader.id && l[2] === 0);
    const channel = structuredClone(source.nodes.find(n => n.id === edge[3]));
    const nodeIds = [loader.id, channel.id];
    loader.outputs.forEach((o, i) => {o.links = i === 0 ? [edge[0]] : [];});
    channel.outputs.forEach(o => {o.links = [];});
    const previewId = Math.max(...nodeIds) + 1;
    const preview = {id: previewId, type: 'PreviewImage', pos: [900, 100], size: [300, 300],
        flags: {}, mode: 0, order: 2, inputs: [{name: 'images', type: 'IMAGE', link: edge[0]+1}],
        outputs: [], properties: {'Node name for S&R': 'PreviewImage'}, widgets_values: []};
    loader.outputs[0].links.push(edge[0]+1);
    const workflow = {last_node_id: previewId, last_link_id: edge[0]+1, version: .4,
        nodes: [loader, channel, preview], links: [edge, [edge[0]+1,loader.id,0,previewId,0,'IMAGE']],
        groups: [], config: {}, extra: {}};
    await context(() => app.loadGraphData(workflow));
    const find = id => context(() => app.graph.getNodeById(id));
    const imageWidget = () => find(loader.id).widgets.find(w => w.name === 'image');
    check(find(loader.id)?.type === 'SwwanLoadAndResizeImage', 'exact v5 loader imported');
    check(imageWidget().value === 'selection.png', 'missing filename preserved, not replaced');
    check(find(channel.id).inputs[0].link === edge[0], 'image-to-mask connection preserved');
    const first = JSON.parse(JSON.stringify(context(() => app.graph.serialize())));
    await context(() => app.loadGraphData(first));
    check(imageWidget().value === 'selection.png', 'missing filename survives save/reload');
    const missing = await context(() => app.graphToPrompt());
    check(missing.output[String(loader.id)].inputs.image === 'selection.png', 'prompt keeps original filename');
    const rejected = await fetch('/prompt', {method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({prompt: missing.output, client_id: 'swwan-upload-widget-regression'})});
    const error = await rejected.json();
    check(rejected.status === 400 && JSON.stringify(error).includes('Invalid image file: selection.png'),
          'backend rejects missing image at execution boundary');
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 8;
    const ctx = canvas.getContext('2d');ctx.fillStyle = '#fff';ctx.fillRect(0,0,8,8);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    const form = new FormData();form.append('image',blob,'selected-mask.png');
    const upload = await fetch('/upload/image', {method:'POST',body:form});
    const uploaded = await upload.json();check(upload.ok,'replacement mask uploaded');
    const name = uploaded.subfolder ? uploaded.subfolder+'/'+uploaded.name : uploaded.name;
    // The dropdown's initial inventory is stale until refresh; the upload widget
    // legitimately adds a new filename without re-registering the node schema.
    if (Array.isArray(imageWidget().options?.values)) imageWidget().options.values.push(name);
    imageWidget().value = name;imageWidget().callback?.(name);
    const selected = JSON.parse(JSON.stringify(context(() => app.graph.serialize())));
    await context(() => app.loadGraphData(selected));
    check(imageWidget().value === name, 'new upload survives save/reload with stale schema inventory');
    const prompt = await context(() => app.graphToPrompt());
    check(prompt.output[String(loader.id)].inputs.image === name,'new upload used by exported prompt');
    return {passed:true,checks,workflow:selected};
})
