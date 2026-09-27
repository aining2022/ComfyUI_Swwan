// Evaluate through Playwright on an isolated ComfyUI CPU service. Run once
// normally and once with swwan_color_widget.js fulfilled as an empty module.
(async () => {
    const {app}=await import('/scripts/app.js');
    const {api}=await import('/scripts/api.js');
    const check=(condition,message)=>{if(!condition)throw Error(message);};
    LiteGraph.namedValuesRestore=false; // prove the repository's own restore
    app.graph.clear();
    const make=kind=>{const n=LiteGraph.createNode(kind);check(n,kind);app.graph.add(n);return n;};
    const widget=(n,name)=>n.widgets?.find(w=>w.name===name);
    const set=(n,name,value)=>{const w=widget(n,name);check(w,`Missing ${name}`);w.value=value;w.callback?.(value);};
    const connect=(src,slot,dst,name)=>{const index=dst.inputs.findIndex(i=>i.name===name);check(index>=0,`Missing ${name} input`);check(src.connect(slot,dst,index),`Connect ${name}`);};
    const image=make('LayerUtility: ColorImage (Swwan)');set(image,'width',32);set(image,'height',24);set(image,'color','#000000');
    const mask=make('SolidMask');set(mask,'width',32);set(mask,'height',24);set(mask,'value',.5);
    const draw=make('SwwanDrawMaskOnImage');connect(image,0,draw,'image');connect(mask,0,draw,'mask');
    draw.onConfigure({widgets_values_named:{color:'255, 255, 255',opacity:1.,device:'cpu'}});
    const resize=make('ImageResizeKJv2Alternative');connect(draw,0,resize,'image');
    const resizeFields={width:768,height:768,upscale_method:'nearest-exact',keep_proportion:'stretch',pad_color:'0, 0, 0',crop_position:'center',divisible_by:0,device:'cpu',resize_mode:'essentials',size_rule:'按长边等比例',edge_length:1024,execute_condition:'总是',edit_fit:'裁剪',fill_color:'#123456',aspect_ratio:'original',proportional_width:1,proportional_height:1,aspect_fit:'letterbox',aspect_method:'lanczos',aspect_round:'8',aspect_scale_side:'longest',aspect_length:1024,essentials_method:'keep proportion',essentials_condition:'always',essentials_interpolation:'lanczos'};
    resize.onConfigure({widgets_values_named:resizeFields});
    const matte=make('SwwanImageMatte');connect(image,0,matte,'image');connect(mask,0,matte,'mask');
    const matteFields={fill_holes:false,crop_mask:false,crop_factor:1.2,stroke_width:0,background_color:'#ffffff'};
    matte.onConfigure({widgets_values_named:matteFields});
    const linked=make('SwwanImageMatte');connect(image,0,linked,'image');connect(mask,0,linked,'mask');linked.onConfigure({widgets_values_named:matteFields});
    const color=make('SwwanColorConverter');set(color,'色值','#123456');connect(color,1,linked,'background_color');
    const save=make('SaveImage');connect(matte,1,save,'images');set(save,'filename_prefix','face_repair_frontend');
    const resizeSave=make('SaveImage');connect(resize,0,resizeSave,'images');set(resizeSave,'filename_prefix','face_repair_resize');
    const linkedSave=make('SaveImage');connect(linked,1,linkedSave,'images');set(linkedSave,'filename_prefix','face_repair_linked');
    const width=make('MathExpression_UTK');set(width,'expression','768');connect(width,0,resize,'width');
    const ids={matte:matte.id,resize:resize.id,draw:draw.id,linked:linked.id,saves:[save.id,resizeSave.id,linkedSave.id]};
    const first=await app.graphToPrompt();
    check(first.output[matte.id].inputs.background_color==='#ffffff','Lost absent color literal');
    check(Array.isArray(first.output[linked.id].inputs.background_color),'Color link overridden');
    check(Array.isArray(first.output[resize.id].inputs.width),'Converted width not linked');
    check(first.output[resize.id].inputs.essentials_interpolation==='lanczos','Resize shifted');
    check(first.output[draw.id].inputs.opacity===1,'Draw shifted');
    const graph=JSON.parse(JSON.stringify(app.graph.serialize()));
    check(graph.nodes.find(n=>String(n.id)===String(ids.matte)).widgets_values_named.background_color==='#ffffff','Color absent from saved map');
    await app.loadGraphData(graph);
    const second=await app.graphToPrompt();
    check(JSON.stringify(first.output)===JSON.stringify(second.output),'Execution inputs changed after save/reload');
    const graphAgain=JSON.parse(JSON.stringify(app.graph.serialize()));
    check(graphAgain.nodes.find(n=>String(n.id)===String(ids.resize)).widgets_values_named.essentials_interpolation==='lanczos','Resize name lost');
    const queued=await api.queuePrompt(0,second);
    return {status:'PASS',color_widget_present:!!widget(app.graph.getNodeById(ids.matte),'background_color'),nodes:graph.nodes.length,links:graph.links.length,checks:['named load/save with native named restoration disabled','nondefault color literal when widget absent','linked color precedence','Draw opacity and CPU','Essentials 768 keep proportion lanczos divisible_by0','same execution parameters after reload'],prompt_id:queued.prompt_id,save_ids:ids.saves};
})()
