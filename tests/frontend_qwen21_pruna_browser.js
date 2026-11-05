// SPDX-License-Identifier: MIT
// Evaluate on the isolated ComfyUI page; never queue model inference.
(async (workflow) => {
    const {app} = await import(new URL('scripts/app.js',document.baseURI).href);
    const vue = [...document.querySelectorAll('div')].find(el=>el.__vue_app__)?.__vue_app__;
    const context = fn => vue ? vue.runWithContext(fn) : fn();
    const checks=[];
    const check=(ok,name)=>{if(!ok) throw Error(name);checks.push(name);};
    await context(()=>app.loadGraphData(workflow));
    const find=id=>context(()=>app.graph.getNodeById(id));
    const widget=(n,key)=>n.widgets.find(w=>w.name===key);
    const set=(n,key,v)=>{const w=widget(n,key);w.value=v;w.callback?.(v);};
    check(context(()=>app.graph._nodes.length)===25,'all 25 UI nodes loaded');
    check(!context(()=>app.graph._nodes.some(n=>n.type==='unknown')),'no unknown nodes');
    check(!find(600).widgets.some(w=>w.name==='control_after_generate'),'no implicit seed randomizer');
    const first=await context(()=>app.graphToPrompt());
    check(first.output['525'].inputs.latent_image[0]==='605','encoder latent -> sampler');
    check(first.output['524'].inputs.noise_seed[0]==='600','shared seed connection');
    check(first.output['610'].inputs.image[0]==='609','save consumes lazy route');
    check(first.output['600'].inputs.qwen_min_resolution===128 && first.output['600'].inputs.qwen_max_resolution===1024,'128/1024 defaults');
    for(const task of ['edit','outpaint','cutout','watermark','face_swap','photoreal','remove','angles']) {
        let n=find(600);set(n,'task',task);
        set(n,'extra_prompt',`test-${task}`);
        if(task==='photoreal') set(n,'generation_mode','text');
        if(task==='remove') set(n,'enhanced',true);
        if(task==='face_swap') set(n,'skin_match',true);
        if(task==='outpaint') set(n,'left',57);
        const saved=JSON.parse(JSON.stringify(context(()=>app.graph.serialize())));
        await context(()=>app.loadGraphData(saved));
        n=find(600);
        check(widget(n,'task').value===task,`task save/reload ${task}`);
        const prompt=await context(()=>app.graphToPrompt());
        check(prompt.output['600'].inputs.task===task && prompt.output['600'].inputs.extra_prompt===`test-${task}`,`API values ${task}`);
        const taskLoras=n.widgets.filter(w=>w.name.startsWith('task_lora_') && !w.hidden);
        check(taskLoras.length===1 && taskLoras[0].name===`task_lora_${task}`,`only active LoRA visible ${task}`);
        if(task==='outpaint') check(!widget(n,'left').hidden && widget(n,'left').value===57,'outpaint precision persisted');
        if(task==='photoreal') check(!widget(n,'output_width').hidden && widget(n,'qwen_max_resolution').hidden && find(601).widgets.every(w=>w.hidden),'text mode hides unused image controls');
        if(task==='face_swap') check(!widget(n,'skin_strength').hidden && widget(n,'color_mode').hidden,'BFS-specific color controls');
    }
    const n=find(600);set(n,'task','outpaint');check(widget(n,'left').value===57,'hidden value retained across all tasks');
    await context(()=>app.loadGraphData(workflow));
    return {passed:true,checks,api:first.output};
})
