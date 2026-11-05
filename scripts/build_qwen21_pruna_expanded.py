# SPDX-License-Identifier: MIT
"""Build visible, ordinary-node Pruna task graphs. Never execute models."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

SCALARS = {'INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COLORCODE', 'COMBO'}
TASKS = ('edit', 'remove', 'watermark', 'outpaint', 'cutout', 'photoreal-image', 'photoreal-text', 'angles')
LABELS = {'edit': '编辑 / 整图或局部', 'remove': '移除 / 普通或增强', 'watermark': '去水印 / 局部或整图',
          'outpaint': '扩图 / 原画布保护', 'cutout': '抠图 / 原生透明补全', 'photoreal-image': '写实图生图',
          'photoreal-text': '写实文生图', 'angles': '多角度', 'face_swap': 'BFS / 展开检测与逐人循环'}


def fields(schema):
    return {k: v for group in ('required', 'optional') for k, v in schema.get('input', {}).get(group, {}).items()}


def default(spec):
    kind, opts = spec[0], spec[1] if len(spec) > 1 else {}
    if 'default' in opts:
        return opts['default']
    if isinstance(kind, list):
        return kind[0] if kind else ''
    if kind == 'COMBO':
        return opts['options'][0]
    return {'STRING': '', 'INT': 0, 'FLOAT': 0., 'BOOLEAN': False, 'COLORCODE': '#ffffff'}.get(kind)


class Graph:
    def __init__(self, schemas, task):
        self.schemas, self.task, self.nodes, self.links = schemas, task, {}, []
        self.used = set()

    def add(self, kind, title, px, py, **values):
        schema = self.schemas[kind]
        self.used.add(kind)
        node_id = len(self.nodes) + 1
        saved, inputs = {}, []
        for name, spec in fields(schema).items():
            typ, opts = spec[0], spec[1] if len(spec) > 1 else {}
            if typ == 'COMFY_AUTOGROW_V3':
                continue
            if isinstance(typ, list) or typ in SCALARS:
                if not opts.get('forceInput'):
                    saved[name] = default(spec)
                    continue
            inputs.append({'name': name, 'type': typ, 'link': None,
                           **({'shape': 7} if name in schema.get('input', {}).get('optional', {}) else {})})
        for name, value in values.items():
            if name not in saved:
                raise ValueError(f'{kind}: not a scalar widget: {name}')
            saved[name] = value
        widgets = []
        for name, value in saved.items():
            widgets.append(value)
            opts = fields(schema)[name][1] if len(fields(schema)[name]) > 1 else {}
            if opts.get('control_after_generate'):
                widgets.append('fixed')
        if kind in ('LoadImage', 'SwwanLoadAndResizeImage'):
            widgets.append('image')
        self.nodes[node_id] = dict(id=node_id, type=kind, title=title, pos=[px,py], size=[370,max(110, 80+len(saved)*28)],
            flags={}, order=node_id-1, mode=0, inputs=inputs,
            outputs=[{'name': name, 'type': typ, 'links': [], 'slot_index': index} for index,(name,typ) in enumerate(zip(schema.get('output_name',schema['output']),schema['output']))],
            properties={'Node name for S&R':kind}, widgets_values=widgets, widgets_values_named=saved)
        if kind in ('PrimitiveStringMultiline','TextEncodeQwenImage21','EditTextEncode_EditUtils'):
            self.nodes[node_id]['size'][1] = max(230,self.nodes[node_id]['size'][1])
        return node_id

    def connect(self, source, target, name):
        sid, output = source if isinstance(source, tuple) else (source,0)
        out = self.nodes[sid]['outputs'][output]
        n = self.nodes[target]
        port = next((p for p in n['inputs'] if p['name']==name), None)
        if port is None:
            port={'name':name,'type':out['type'],'link':None}
            if name in n['widgets_values_named']:
                port['widget']={'name':name}
            n['inputs'].append(port)
        if port['link'] is not None:
            raise ValueError(f'Duplicate {target}.{name}')
        if port['type'] not in ('*','ANY',out['type']) and out['type'] not in ('*','ANY'):
            raise ValueError(f'Incompatible {sid} -> {target}.{name}: {out["type"]}/{port["type"]}')
        lid=len(self.links)+1
        self.links.append([lid,sid,output,target,n['inputs'].index(port),out['type']])
        port['link']=lid;out['links'].append(lid)

    def image_preview(self, source, title, x, y):
        n=self.add('PreviewImage',title,x,y); self.nodes[n]['size']=[370,300]; self.connect(source,n,'images');return n

    def mask_preview(self, source, title, x, y):
        n=self.add('MaskToImage',title+' / mask→image',x,y);self.connect(source,n,'mask')
        self.image_preview(n,title,x,y+135)

    def value_preview(self, source, title, x, y):
        n=self.add('PreviewAny',title,x,y);self.connect(source,n,'source');return n

    def switch(self, flag, yes, no, title, x,y):
        n=self.add('easy ifElse',title,x,y,boolean=False)
        self.connect(flag,n,'boolean');self.connect(yes,n,'on_true');self.connect(no,n,'on_false');return n

    def note(self, text, x=0,y=-450):
        n=len(self.nodes)+1
        self.nodes[n]=dict(id=n,type='MarkdownNote',title='阅读顺序与边界 / '+LABELS[self.task],pos=[x,y],size=[1150,370],flags={},order=n-1,mode=0,inputs=[],outputs=[],properties={},widgets_values=[text])

    def finish(self):
        # Keep every socket/widget visible, with no overlapping node rectangles.
        columns = {}
        for node in self.nodes.values():
            if node['type'] == 'MarkdownNote':
                node['pos'][1] = -700
                node['size'] = [2100, 570]
                continue
            node['size'][1] = max(node['size'][1], 90 + 28 * len(node.get('widgets_values_named', {}))
                                  + 22 * max(len(node['inputs']), len(node['outputs'])))
            columns.setdefault(node['pos'][0], []).append(node)
        for column in columns.values():
            bottom = -1
            for node in sorted(column, key=lambda n: (n['pos'][1], n['id'])):
                node['pos'][1] = max(node['pos'][1], bottom + 80)
                bottom = node['pos'][1] + node['size'][1]
        w=dict(last_node_id=max(self.nodes),last_link_id=len(self.links),nodes=list(self.nodes.values()),links=self.links,
               groups=[],config={},extra={'ds':{'scale':.35,'offset':[100,500]},'swwan_expanded_version':5,'task':self.task,
               'validation':'Static and CPU contracts only; this expanded wiring has not passed GPU acceptance.'},version=.4)
        for start, end, title in ((0, 2480, '01 / 输入、权限、模式与模型'),
                                  (2500, 3340, '02 / 同源裁剪与面积预算'),
                                  (3350, 4600, '03 / 标记、提示词与输入预览'),
                                  (4610, 6680, '04 / 原生编码与六步采样'),
                                  (6700, 9230, '05 / 校色、几何、独立贴回与alpha')):
            members = [n for n in self.nodes.values() if n['type'] != 'MarkdownNote' and start <= n['pos'][0] < end]
            if members:
                bottom = max(n['pos'][1] + n['size'][1] for n in members)
                w['groups'].append(dict(id=len(w['groups'])+1, title=title,
                    bounding=[start-30, -65, end-start+10, bottom+115], color='#38655e', font_size=24, flags={}))
        validate(w,self.schemas)
        return w


def source_values(source, node_id):
    return next(n for n in source['nodes'] if n['id']==node_id)['widgets_values_named']


def model_chain(g, source, task):
    unet=g.add('UNETLoader','原v3 INT8模型',0,1700,**source_values(source,509))
    pruna=g.add('LoraLoaderModelOnly','Pruna v0.2 / 权重1',420,1700,**source_values(source,527));g.connect(unet,pruna,'model')
    selected=pruna
    if task=='edit':
        extra=g.add('LoraLoaderModelOnly','原v3附加LoRA / 只在edit使用',840,1700,**source_values(source,530));g.connect(pruna,extra,'model')
        flag=g.add('PrimitiveBoolean','启用原v3附加LoRA / 工具对照请关闭',420,1990,value=True)
        selected=g.switch(flag,extra,pruna,'附加LoRA显式开关',1260,1700)
    elif task in ('remove','watermark','outpaint','cutout','photoreal-image','photoreal-text','angles'):
        index={'outpaint':2,'cutout':3,'watermark':4,'photoreal-image':6,'photoreal-text':6,'remove':7,'angles':8}[task]
        strength={2:1.,3:1.,4:.8,6:.8,7:1.,8:.9}[index]
        selected=g.add('LoraLoaderModelOnly',LABELS[task]+' / 独立任务LoRA',840,1700,lora_name=source_values(source,566)[f'lora_{index}_name'],strength_model=strength);g.connect(pruna,selected,'model')
    att=g.add('ModelAttentionBackend','原v3 Attention / 不加Cache或Spectrum',1680,1700,**source_values(source,532));g.connect(selected,att,'model')
    clip=g.add('CLIPLoader','原v3 INT8 CLIP',0,2000,**source_values(source,548))
    vae=g.add('VAELoader','原v3 Qwen 2.1 VAE',0,2250,**source_values(source,511))
    return att,clip,vae


def exact_mask(g,title,x,y):
    n=g.add('SwwanLoadAndResizeImage',title+' / Pillow读取，不缩放',x,y,image='selection.png',resize=False,background_color='',mask_channel='red')
    channel=g.add('ImageToMask',title+' / 显式红通道，不使用alpha',x+420,y,channel='red');g.connect(n,channel,'image')
    return channel


def crop(g,image,mask,x,y,backend='swwan',factor=1.3):
    n=g.add('SwwanCropByMaskV5','局部裁剪 / BlueBox / 128–1024面积预算',x,y,crop_mode='qwen21_aligned',
            fill_mask_holes=False,edit_expansion_mode='factor',edit_top_factor=factor,edit_bottom_factor=factor,
            edit_left_factor=factor,edit_right_factor=factor,qwen_resolution_mode='crop_area',qwen_min_resolution=128,
            qwen_max_resolution=1024,qwen_restore_backend=backend,qwen_tolerance=1.)
    g.connect(image,n,'image');g.connect(mask,n,'mask')
    for slot,title,dy in [(9,'原生裁剪宽',0),(10,'原生裁剪高',140),(11,'实际面积预算R',280),(6,'推理画布宽',420),(7,'推理画布高',560)]:
        g.value_preview((n,slot),title,x+420,y+dy)
    return n


def composite(g,dest,source,mask,title,x,y,xy=None):
    n=g.add('ImageCompositeMasked',title,x,y,x=0,y=0,resize_source=False)
    g.connect(dest,n,'destination');g.connect(source,n,'source')
    if mask is not None:g.connect(mask,n,'mask')
    if xy:
        g.connect(xy[0],n,'x');g.connect(xy[1],n,'y')
    return n


def solid(g,size,value,title,x,y):
    n=g.add('SolidMask',title,x,y,value=value,width=512,height=512);g.connect((size,0),n,'width');g.connect((size,1),n,'height');return n


def build_task(schemas,source,task):
    g=Graph(schemas,task)
    model,clip,vae=model_chain(g,source,task)
    text_only=task=='photoreal-text'
    prompt_id={'edit':555,'outpaint':553,'cutout':563,'watermark':567,'remove':556,'photoreal-image':568,'photoreal-text':568,'angles':578}[task]
    prompt=source_values(source,prompt_id)['value']
    if task=='angles':prompt='<mva> right side view, high-angle shot'
    if task=='cutout':prompt=prompt.replace('输出透明背景PNG','移除红色定位标记，输出透明背景PNG')
    base=g.add('PrimitiveStringMultiline','固定触发提示词 / 可直接检查修改',3360,1000,value=prompt)
    extra=g.add('PrimitiveStringMultiline','额外指令 / 各任务独立，不继承其他任务',3360,1300,value='')
    text=g.add('StringConcatenate','触发词 + 额外指令',3780,1100,string_a='',string_b='',delimiter='\n');g.connect(base,text,'string_a');g.connect(extra,text,'string_b')
    g.value_preview(text,'最终实际提示词',4200,1100)
    prepared=mask=None
    if not text_only:
        loader=g.add('LoadImage','原图 / 保留实际ComfyUI解码基线',0,0,image=source_values(source,551)['image']);g.nodes[loader]['size']=[370,450]
        size=g.add('GetImageSize','原图尺寸',420,0);g.connect(loader,size,'image')
        protection = None
        if task in ('edit','remove','watermark'):
            protection=solid(g,size,0.,'保护区 / 默认全黑；可接独立红通道mask',420,170)
        image=loader
        local=task in ('edit','remove','watermark','cutout')
        if local:
            selection=exact_mask(g,'选区文件',0,530)
            if task!='cutout':
                minus=g.add('SwwanMaskCombine','原坐标选区 - 保护',840,530,operation='相减',bbox_mode='关闭',alignment='居中');g.connect(selection,minus,'mask_1');g.connect(protection,minus,'mask_2');selection=minus
            fill=g.add('SwwanMaskProcess','仅圈选时填洞 / 普通软mask旁路',840,840,operation='fill_holes')
            g.connect(selection,fill,'mask')
            circle=g.add('PrimitiveBoolean','闭合圈选 / false为普通涂抹',420,1000,value=False)
            selected=g.switch(circle,fill,selection,'圈选填洞 / 涂抹软值',1260,530)
            if task!='cutout':
                minus2=g.add('SwwanMaskCombine','填洞后再次扣保护',1680,530,operation='相减',bbox_mode='关闭',alignment='居中');g.connect(selected,minus2,'mask_1');g.connect(protection,minus2,'mask_2');selected=minus2
            if task in ('edit','watermark'):
                white=solid(g,size,1.,'整图编辑范围',840,0)
                full=g.add('SwwanMaskCombine','整图范围 - 保护',1260,0,operation='相减',bbox_mode='关闭',alignment='居中');g.connect(white,full,'mask_1');g.connect(protection,full,'mask_2')
                flag=g.add('PrimitiveBoolean','局部模式 / 关闭为整图',1260,250,value=task=='watermark')
                selected=g.switch(flag,selected,full,'局部选区 / 整图范围',2100,530)
            if task=='remove':
                block=g.add('SwwanBlockifyMask','增强分支 / 原图坐标32px块化',1260,900,block_size=32,device='cpu');g.connect(selected,block,'masks')
                again=g.add('SwwanMaskCombine','块化后再次扣保护',1680,900,operation='相减',bbox_mode='关闭',alignment='居中');g.connect(block,again,'mask_1');g.connect(protection,again,'mask_2')
                enhanced=g.add('PrimitiveBoolean','增强模式 / 普通false保持40%绿色',1260,1200,value=False)
                selected=g.switch(enhanced,again,selected,'普通 / 增强有效范围',2100,530)
                opacity1=g.add('PrimitiveFloat','普通覆盖0.4',1680,1200,value=.4);opacity2=g.add('PrimitiveFloat','增强覆盖1.0',1680,1350,value=1.)
                opacity=g.switch(enhanced,opacity2,opacity1,'当前绿色不透明度',2100,1150)
            mask=selected
        elif task=='outpaint':
            pad=g.add('SwwanImagePadKJ','四向扩图 / 灰色画布 / 原图不缩放',840,0,left=128,right=128,top=0,bottom=0,extra_padding=0,pad_mode='color',color='128,128,128');g.connect(loader,pad,'image');image=pad
            mask=(pad,1)
            image_size=g.add('GetImageSize','扩图后画布尺寸',1260,0);g.connect(pad,image_size,'image')
            # Grow is an explicit permission change, not an implicit feather operation.
            grow=g.add('GrowMask','接缝带 / 0严格保留旧画布',1680,0,expand=0,tapered_corners=False);g.connect(mask,grow,'mask');mask=grow
        else:
            mask=solid(g,size,1.,'整图范围 / 不强制局部贴回',840,0)
        crop_mask=mask
        if task=='outpaint':crop_mask=solid(g,image_size,1.,'整张扩图画布参与预算 / 不四次拆图',2100,0)
        backend='goohai' if task in ('remove','watermark','outpaint') else 'swwan'
        cp=crop(g,image,crop_mask,2520,0,backend=backend,factor=1. if task=='outpaint' else 1.3)
        prepared=cp
        if task=='remove':
            mark=g.add('SwwanDrawMaskOnImage','只在副本绘制绿色 / 干净裁剪保留',3360,0,color='0,255,0',opacity=.4,device='cpu');g.connect(cp,mark,'image');g.connect((cp,5),mark,'mask');g.connect(opacity,mark,'opacity');prepared=mark
        if task=='cutout':
            erode=g.add('GrowMask','红色内轮廓 / 向内收缩2px',2940,800,expand=-2,tapered_corners=True);g.connect((cp,5),erode,'mask')
            ring=g.add('SwwanMaskCombine','原mask - 腐蚀mask = 内轮廓',3360,750,operation='相减',bbox_mode='关闭',alignment='居中');g.connect((cp,5),ring,'mask_1');g.connect(erode,ring,'mask_2')
            mark=g.add('SwwanDrawMaskOnImage','红色定位轮廓 / 不覆盖整块主体',3780,0,color='255,0,0',opacity=1.,device='cpu');g.connect(cp,mark,'image');g.connect(ring,mark,'mask');prepared=mark
        g.image_preview(cp,'干净裁剪 / 校色参考',2940,1450)
        g.image_preview(prepared,'实际模型输入',3780,500)
        g.mask_preview(mask,'原坐标有效范围',2100,1450)

    if task=='cutout':
        mc=g.add('QwenImage21ModelConfig_EditUtils','原生EditUtils模型配置',4200,0,instruction='')
        longest=g.add('easy imageSizeByLongerSide','已对齐图实际长边 / 禁止第二次缩放',4200,230);g.connect(prepared,longest,'image')
        conf=g.add('QwenImage21ConfigPreparer_EditUtils','抠图编码配置 / 图像与padding可检查',4620,0,to_ref=False,ref_main_image=True,ref_longest_edge=1024,ref_crop='pad',ref_upscale='lanczos',to_vl=True,ref_resize_mode='longest_edge',rope_x_offset=0,rope_y_offset=0)
        g.connect(prepared,conf,'image');g.connect(longest,conf,'ref_longest_edge')
        enc=g.add('EditTextEncode_EditUtils','原生EditUtils编码器',5040,0,prompt='');g.connect(mc,enc,'model_config');g.connect(conf,enc,'configs')
        neg=g.add('ConditioningZeroOut','负向清零 / CFG1',5460,300);g.connect(enc,neg,'conditioning');negative=neg;latent=(enc,1)
    else:
        enc=g.add('TextEncodeQwenImage21','官方编码 / resolution=0 / 同源latent',4620,0,prompt='',negative_prompt='',resolution=0)
        if not text_only:g.connect(prepared,enc,'images.image_1')
        negative=(enc,1);latent=(enc,2)
        if task=='edit':
            ref=g.add('LoadImage','可选参考图 / 默认禁用；启用后接编码器image_2',3780,1700,image='reference.png');g.nodes[ref]['mode']=2
            # Explicit spare port; no stale reference is silently encoded.
            g.nodes[enc]['inputs'].append({'name':'images.image_2','type':'IMAGE','shape':7,'link':None})
        if text_only:
            g.value_preview(latent,'官方纯文本latent / 默认1024×1024，不用SD空latent',5040,600)
    g.connect(clip,enc,'clip');g.connect(vae,enc,'vae');g.connect(text,enc,'prompt')
    guider=g.add('CFGGuider','原v3 CFG=1',5460,0,cfg=1.);g.connect(model,guider,'model');g.connect(enc,guider,'positive');g.connect(negative,guider,'negative')
    noise=g.add('RandomNoise','固定seed / 手动比较',5040,1000,noise_seed=source_values(source,524)['noise_seed'])
    sampler=g.add('KSamplerSelect','原v3 Euler',5460,1000,sampler_name='euler')
    sigmas=g.add('ManualSigmas','原v3 Pruna 6步 / 完整sigmas',5460,1200,**source_values(source,522))
    sample=g.add('SamplerCustomAdvanced','原生采样器 / 不隐藏内部调用',5880,0)
    for src,name in [(noise,'noise'),(guider,'guider'),(sampler,'sampler'),(sigmas,'sigmas'),(latent,'latent_image')]:g.connect(src,sample,name)
    decode=g.add('VAEDecode','VAE原始解码 / 尚未校色贴回',6300,0);g.connect(sample,decode,'samples');g.connect(vae,decode,'vae')
    g.image_preview(decode,'原始生成块 / RGB或RGBA',6300,550)
    output=decode
    if not text_only:
        if task=='cutout':
            unpad=g.add('CropWithPadInfo_EditUtils','只去编码器padding / 保留生成RGBA',6720,0);g.connect((enc,5),unpad,'pad_info');g.connect(decode,unpad,'image')
            unpack=g.add('SwwanQwenGeometryUnpack','只反变换局部几何 / 不做贴回',7140,0,require_alpha=True);g.connect(unpad,unpack,'image');g.connect((cp,4),unpack,'seam')
            split=g.add('SplitImageWithAlpha','生成RGBA分离 / MASK=1-alpha',7560,0);g.connect(unpack,split,'image')
            white=solid(g,size,1.,'原尺寸全透明底 / inverse alpha=1',7140,650)
            empty=g.add('EmptyImage','原尺寸空白RGB画布',7140,1000,width=512,height=512,batch_size=1,color=0);g.connect((size,0),empty,'width');g.connect((size,1),empty,'height')
            rgb=composite(g,empty,split,None,'RGB按还原坐标贴回 / 不用用户选区截补全',7980,0,xy=((unpack,1),(unpack,2)))
            al=g.add('MaskComposite','生成inverse alpha贴回透明画布 / 不叠乘alpha',7980,550,x=0,y=0,operation='multiply');g.connect(white,al,'destination');g.connect((split,1),al,'source');g.connect((unpack,1),al,'x');g.connect((unpack,2),al,'y')
            output=g.add('JoinImageWithAlpha','RGB + 生成alpha / 原尺寸透明输出',8400,0);g.connect(rgb,output,'image');g.connect(al,output,'alpha')
        else:
            split=g.add('SplitImageWithAlpha','后处理只取生成RGB / 原alpha另行恢复',6720,0);g.connect(decode,split,'image')
            if backend=='goohai':
                restore=g.add('GH_CropRestore','孤海恢复 / 显式校色强度、扩展、羽化',7140,0,遮罩扩展百分比=4,遮罩羽化百分比=5,颜色还原强度=1.)
                g.connect((cp,4),restore,'接缝');g.connect(split,restore,'裁剪图像');g.connect((cp,5),restore,'裁剪遮罩');g.connect(image,restore,'背景图像')
            else:
                calibrated=g.add('ColorCalibratorBlueBox','BlueBox校色 / 强度0旁路，改色勿开全局校色',7140,0,threshold=.08,expand=64,open_k=13,close_k=51,strength=0.,outside_mode='edited',feather=24)
                g.connect(cp,calibrated,'image_original');g.connect(split,calibrated,'image_edited');g.connect((cp,5),calibrated,'mask')
                restore=g.add('SwwanRestoreCropBoxV4','原生SEAM恢复 / 不猜坐标，不重复逆变换',7560,0,feathering=0,device='CPU',expand_percent=0,feather_percent=0)
                g.connect(image,restore,'background_image');g.connect(calibrated,restore,'croped_image');g.connect((cp,2),restore,'crop_box');g.connect((cp,4),restore,'region_info')
            g.image_preview(restore,'恢复与校色之后 / 最终门控之前',7560,1000)
            binary=g.add('ThresholdMask','最终权限仅取有效mask非零支持域',7560,550,value=0.);g.connect(mask,binary,'mask')
            gated=composite(g,image,restore,binary,'最终门控 / 范围外取回原像素',7980,0)
            # ComfyUI's inverse alpha is preserved independently of RGB generation.
            inv=(loader,1)
            if task=='outpaint':
                alpha_rgb=g.add('MaskToImage','原图inverse alpha→图像',6720,1400);g.connect(inv,alpha_rgb,'mask')
                alpha_size=g.add('ImageScale','alpha恢复原图尺寸 / RGB无alpha时兼容64×64空mask',6720,1750,upscale_method='nearest-exact',width=512,height=512,crop='disabled')
                g.connect(alpha_rgb,alpha_size,'image');g.connect((size,0),alpha_size,'width');g.connect((size,1),alpha_size,'height')
                alpha_pad=g.add('SwwanImagePadKJ','alpha同四向扩展 / 新区域不透明',7140,1400,left=128,right=128,top=0,bottom=0,extra_padding=0,pad_mode='color',color='0,0,0');g.connect(alpha_size,alpha_pad,'image')
                for name in ('left','right','top','bottom'):
                    value=g.nodes[pad]['widgets_values_named'][name]
                    primitive=g.add('PrimitiveInt','扩图'+name+' / 同时驱动RGB和alpha',0 if name in ('left','right') else 420,2700+(0 if name in ('left','top') else 160),value=value)
                    g.connect(primitive,pad,name);g.connect(primitive,alpha_pad,name)
                to_mask=g.add('ImageToMask','扩展后的inverse alpha',7560,1500,channel='red');g.connect(alpha_pad,to_mask,'image');inv=to_mask
            output=g.add('JoinImageWithAlpha','恢复原图alpha / RGB时为全不透明',8400,0);g.connect(gated,output,'image');g.connect(inv,output,'alpha')
    save=g.add('SwwanSaveImage','最终保存 / PNG直通alpha',8820,0,file_format='png',filename_prefix='Pruna_v5_expanded/'+task,input_mode='straight',alpha_mode='auto',save_workflow_as_json=True)
    g.connect(output,save,'image');g.image_preview(output,'最终结果 / 对照原图',8820,850)
    note=('## '+LABELS[task]+' / v5 展开版\n\n从左到右逐步检查：输入 → mask → 裁剪/预算 → 标记 → 编码 → 原六步采样 → 解码 → 校色/几何 → 独立门控/alpha → 保存。\n'
          '所有处理节点和连线可见；没有Pruna任务config、聚合节点或隐藏子图。各任务单独使用，不同时运行总览全部行。\n'
          '模型/Pruna六步/CFG1/Euler与v3一致。预算128/1024是面积，不是长边；不重复接Cut BlueBox。\n'
          '保护默认全黑；需要时用已有Pillow加载器IMAGE→ImageToMask(red)替换。文件mask不要接主图alpha；即使整图模式，也请为选区加载器选择有效文件，否则ComfyUI可能预校验失败。圈选填洞不检查是否闭合，请自行查看预览。\n'
          '本版只做静态/CPU合同验证，未重新GPU验收；不能继承v4的效果或速度结论。')
    if text_only:note+='\n纯文生图直接使用官方编码器默认1024×1024 latent，不连接图片；本版本不伪造其他模型的空latent。'
    if task=='angles':note+='\n角度提示词直接可编辑；修正v3误接的绿色移除提示词，使用<mva>角度触发。效果尚未通过。'
    if task=='outpaint':note+='\n四向padding INT同时连接RGB和alpha，单位像素；GrowMask=0严格保护旧画布，正值明确扩大旧图可改范围。'
    if task=='cutout':note+='\nrequire_alpha=true：若VAE未输出真实RGBA则停止报错；不将用户选择mask冒充生成alpha，不用选区截断补全部分。'
    g.note(note)
    w=g.finish();w['extra']['required_nodes']=sorted(g.used)
    return w


def api_from_ui(workflow):
    links={x[0]:x for x in workflow['links']};result={}
    for n in workflow['nodes']:
        if n.get('mode',0)!=0 or n['type'] in ('MarkdownNote','Note'):continue
        inputs=copy.deepcopy(n.get('widgets_values_named',{}))
        for ui_field in ('control_after_generate', 'upload', 'info'):
            inputs.pop(ui_field,None)
        for p in n.get('inputs',[]):
            if p.get('link') is not None:
                link=links[p['link']];inputs[p['name']]=[str(link[1]),link[2]]
        result[str(n['id'])]={'class_type':n['type'],'inputs':inputs,'_meta':{'title':n.get('title',n['type'])}}
    return result


def validate(w,schemas):
    nodes={n['id']:n for n in w['nodes']};links={l[0]:l for l in w['links']}
    if len(nodes)!=len(w['nodes']) or len(links)!=len(w['links']):raise ValueError('Duplicate IDs')
    for lid,s,slot,t,inp,kind in w['links']:
        op,ip=nodes[s]['outputs'][slot],nodes[t]['inputs'][inp]
        if lid not in (op.get('links') or []) or ip.get('link')!=lid:raise ValueError('Broken bidirectional link')
        if op['type'] not in ('*','ANY',ip['type']) and ip['type'] not in ('*','ANY'):
            raise ValueError('Incompatible linked types')
    pending, done = set(nodes), set()
    while pending:
        ready = {n for n in pending if all(l[1] in done for l in w['links'] if l[3] == n)}
        if not ready:raise ValueError('Cyclic workflow')
        done.update(ready);pending.difference_update(ready)
    for n in w['nodes']:
        if n['type'].startswith('SwwanQwen21') or n['type'] in ('SetNode','GetNode'):raise ValueError('Hidden task/virtual wire: '+n['type'])
        if n['type'] in ('MarkdownNote','Note'):continue
        schema=schemas[n['type']];params=fields(schema);ports={p['name']:p for p in n.get('inputs',[])}
        for p in ports.values():
            if p.get('link') is not None and p['link'] not in links:raise ValueError('Dangling input')
        for name,spec in schema.get('input',{}).get('required',{}).items():
            if spec[0]=='COMFY_AUTOGROW_V3':continue
            if name not in n.get('widgets_values_named',{}) and ports.get(name,{}).get('link') is None:raise ValueError(f'Missing {n["type"]}.{name}')
        for name,value in n.get('widgets_values_named',{}).items():
            if name in ('control_after_generate','upload','info'):continue
            if name not in params:raise ValueError(f'Unknown {n["type"]}.{name}')
            spec=params[name];kind=spec[0];opts=spec[1] if len(spec)>1 else {}
            if kind == 'BOOLEAN' and not isinstance(value,bool):raise ValueError(f'Invalid boolean {name}')
            if isinstance(kind,list) and name not in ('image','lora_name','unet_name','clip_name','vae_name') and value not in kind:
                raise ValueError(f'Invalid enum {name}: {value}')
            if kind in ('INT','FLOAT') and not isinstance(kind,list):
                if not isinstance(value,(int,float)) or isinstance(value,bool):raise ValueError(f'Invalid number {name}')
                if not opts.get('min',-float('inf'))<=value<=opts.get('max',float('inf')):raise ValueError(f'Out of range {name}')
    return w


def combine(workflows):
    combined={'last_node_id':0,'last_link_id':0,'nodes':[],'links':[],'groups':[],'config':{},'extra':{'ds':{'scale':.12,'offset':[100,500]},'swwan_expanded_version':5},'version':.4}
    for row,(task,w) in enumerate(workflows.items()):
        offset=combined['last_node_id'];loffset=combined['last_link_id'];copyw=copy.deepcopy(w)
        xs=[n['pos'][0] for n in copyw['nodes']];ys=[n['pos'][1] for n in copyw['nodes']];minx,miny=min(xs),min(ys)
        height=max(n['pos'][1]+n['size'][1] for n in copyw['nodes'])-miny
        ybase=0 if not combined['groups'] else combined['groups'][-1]['bounding'][1]+combined['groups'][-1]['bounding'][3]+250
        for n in copyw['nodes']:
            n['id']+=offset;n['pos']=[n['pos'][0]-minx,n['pos'][1]-miny+ybase]
            n['mode']=n.get('mode',0) if row==0 else 2
            for p in n.get('inputs',[]):
                if p.get('link') is not None:p['link']+=loffset
            for p in n.get('outputs',[]):p['links']=[x+loffset for x in p.get('links') or []]
            combined['nodes'].append(n)
        combined['links'].extend([[lid+loffset,s+offset,slot,t+offset,inp,kind] for lid,s,slot,t,inp,kind in w['links']])
        maxx=max(n['pos'][0]+n['size'][0] for n in copyw['nodes'])
        combined['groups'].append(dict(id=row+1,title=f'{row+1:02d} / {LABELS[task]} / '+('当前启用' if row==0 else '默认禁用，独立文件更方便运行'),bounding=[-50,ybase-60,maxx+100,height+120],color='#3f789e',font_size=26,flags={}))
        combined['last_node_id']=max(n['id'] for n in combined['nodes']);combined['last_link_id']=max(x[0] for x in combined['links'])
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--schemas',type=Path,required=True);p.add_argument('--bfs-source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be a new directory')
    schemas=json.loads(a.schemas.read_text());source=json.loads(a.source.read_text())
    workflows={task:build_task(schemas,source,task) for task in TASKS}
    if a.bfs_source:
        from pruna_expanded_bfs import build_bfs, to_api
        workflows['face_swap']=build_bfs(json.loads(a.bfs_source.read_text()),source,schemas)
    a.output.mkdir(parents=True)
    for task,w in workflows.items():
        (a.output/f'pruna-v5-{task}.json').write_text(json.dumps(w,ensure_ascii=False,indent=2)+'\n')
        api = to_api(w,schemas) if task=='face_swap' else api_from_ui(w)
        (a.output/f'pruna-v5-{task}-api.json').write_text(json.dumps(api,ensure_ascii=False,indent=2)+'\n')
    allw=combine(workflows)
    (a.output/'pruna-v5-expanded-all.json').write_text(json.dumps(allw,ensure_ascii=False,indent=2)+'\n')
    used={n['type'] for w in workflows.values() for n in w['nodes'] if n['type'] not in ('MarkdownNote','Note')}
    snapshot = {k:copy.deepcopy(schemas[k]) for k in sorted(used)}
    for kind,schema in snapshot.items():
        for name,spec in fields(schema).items():
            if name in ('image','lora_name','unet_name','clip_name','vae_name','model_name') and isinstance(spec[0],list):
                # Freeze only referenced assets, not the machine's full model/input inventory.
                values = {n.get('widgets_values_named',{}).get(name) for w in workflows.values()
                          for n in w['nodes'] if n['type']==kind}
                spec[0] = sorted(v for v in values if isinstance(v,str)) or spec[0][:1]
    (a.output/'schemas.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    provenance = {'pruna_source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
                  'bfs_source_sha256':hashlib.sha256(a.bfs_source.read_bytes()).hexdigest() if a.bfs_source else None,
                  'schema_origin':'Saved L20 object_info plus local atomic geometry interface; not live validation',
                  'validation':'Static/CPU only. GPU and browser import/roundtrip acceptance are pending.'}
    (a.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print({task:len(w['nodes']) for task,w in workflows.items()})


if __name__=='__main__':main()
