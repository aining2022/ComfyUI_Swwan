# SPDX-License-Identifier: MIT
"""Pruna-specific contracts, distinct from the historical Turbo dispatch/runtime."""
from ..ops import qwen21_pruna as P


def number(kind, default, minimum, maximum, tip, step=1):
    return (kind, {'default':default, 'min':minimum, 'max':maximum, 'step':step, 'tooltip':tip})


class Qwen21PrunaControls:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            'task': (list(P.TASKS), {'default':'edit','tooltip':'任务联动LoRA、提示词、前后处理；BFS单独惰性分支。'}),
            'extra_prompt': ('STRING', {'default':'HD, enhance, red lipstick, mouth wide open','multiline':True,'tooltip':'用户额外指令；普通编辑必填。切换任务时请清除不适用的内容。'}),
            'negative_prompt': ('STRING', {'default':'','multiline':True,'tooltip':'官方编码器负面提示词；CFG=1不保证负向引导生效。EditUtils路径必须留空。'}),
            'seed': number('INT',482316605808414,0,0xffffffffffffffff,'所有任务共用seed；同seed不承诺跨硬件逐像素一致。'),
            'edit_scope': (['full','local'], {'default':'full','tooltip':'仅edit/watermark；local必须有选区，full处理整图。'}),
            'generation_mode': (['image','text'], {'default':'image','tooltip':'仅photoreal的text跳过所有图片/参考输入。'}),
            'encoder': (['auto','official','editutils'], {'default':'auto','tooltip':'auto: cutout/BFS使用EditUtils，其余官方2.1；缺依赖明确报错。'}),
            'qwen_min_resolution': number('INT',128,32,2048,'面积预算下限，向上32对齐；低于512效果待GPU验收。'),
            'qwen_max_resolution': number('INT',1024,512,2048,'面积预算上限，向下32对齐；不是长边。'),
            'qwen_tolerance': number('FLOAT',1.,.2,3.,'BlueBox候选面积容差，不是内容位移精度。',.1),
            'context_factor': number('FLOAT',1.3,1.,10.,'每侧增加bbox短边*(factor-1)的上下文；不扩大最终编辑权限。',.1),
            'reference_resolution': number('INT',768,128,2048,'身份参考独立面积预算；不套主图BOX。',32),
            'color_mode': (['auto','none','goohai','bluebox'], {'default':'auto','tooltip':'auto仅局部编辑/移除/扩图用GH；默认不全局校色。GH与BlueBox互斥。'}),
            'color_strength': number('FLOAT',1.,0.,1.,'校色强度；不改变alpha。',.05),
            'expand_percent': number('INT',4,0,30,'GH贴回扩展百分比；最终原坐标门控限制修改。'),
            'feather_percent': number('INT',5,0,30,'贴回羽化百分比；最终保护mask不等于羽化权重。'),
            'marking': (['brush','circle'], {'default':'brush','tooltip':'brush保留软mask；circle要求闭合并填内部。'}),
            'enhanced': ('BOOLEAN', {'default':False,'tooltip':'仅remove：块化选区后再次扣保护，绿色覆盖改100%。'}),
            'mark_opacity': number('FLOAT',.4,0.,1.,'普通remove绿色标记不透明度；cutout固定红色内轮廓。',.05),
            'block_size': number('INT',32,8,512,'增强移除的原图选区块预算；不是Qwen32对齐。'),
            'padding_unit': (['pixels','percent'], {'default':'pixels','tooltip':'扩图单位：原图像素，或对应原图宽/高的百分比。'}),
            **{k:number('INT',v,0,4096,'仅outpaint；四向扩展，percent按对应原图轴计算。') for k,v in [('left',128),('right',128),('top',0),('bottom',0)]},
            'seam_band': number('INT',0,0,4096,'outpaint允许修改的原图内侧接缝带像素；0严格保留旧画布。'),
            'people': number('INT',1,1,6,'BFS目标人数，按行从左到右匹配参考图。'),
            'swap_head': ('BOOLEAN', {'default':False,'tooltip':'BFS换头范围开关，false保持换脸流程。'}),
            'forehead': ('BOOLEAN', {'default':False,'tooltip':'BFS原有额头范围处理开关，不是通用mask膨胀。'}),
            'skin_match': ('BOOLEAN', {'default':False,'tooltip':'仅BFS保留原肤色匹配链。'}),
            'skin_strength': number('FLOAT',.25,0.,1.,'仅BFS启用肤色匹配时的强度。',.05),
            **{k:number('INT',1024,32,4096,'仅纯文生图输出画布，32倍数。',32) for k in ('output_width','output_height')},
            'azimuth': number('INT',0,0,330,'仅angles：水平角度。',30),
            'elevation': ([0,30,60,90], {'default':0,'tooltip':'多角度任务的仰俯视角档位，以提示词控制，不是几何投影。'}),
            'close_up': ('BOOLEAN', {'default':False,'tooltip':'多角度任务追加close-up特写提示词。'}),
            'lora_strength': number('FLOAT',-1.,-1.,2.,'-1使用v3任务预设，0禁用当前任务LoRA。',.05),
            'additional_lora_mode': (['off','edit_only','all'], {'default':'edit_only','tooltip':'原v3附加LoRA默认只在edit使用，不污染工具任务。'}),
            'additional_lora_name': ('STRING', {'default':'qwen21/qwen2_1-femaledVaginus-nsfw-v1.3-fine.safetensors','tooltip':'v3附加LoRA路径；不需要时将additional_lora_mode设off。此项不是Pruna加速LoRA。'}),
            'additional_lora_strength': number('FLOAT',.6,0.,2.,'附加LoRA权重，不是Pruna加速权重。',.05),
            **{f'task_lora_{task}':('STRING', {'default':path,'tooltip':'相对于ComfyUI models/loras的任务文件；空表示不加载。'}) for task,path in P.LORAS.items()},
        }
        required['seed'][1]['control_after_generate'] = False
        return {'required':required}
    RETURN_TYPES = ('QWEN21_PRUNA_CONFIG','STRING','INT')
    RETURN_NAMES = ('config','actual_prompt','seed')
    FUNCTION = 'build'
    DESCRIPTION = 'Pruna v3任务配置。模型/6步采样可见；新合同不修改旧Turbo节点。128/1024只限推理面积，不默认缩小最终输出。说明见docs/QWEN21_PRUNA.md。'
    def build(self, **kwargs):
        c=P.resolve(kwargs)
        return c,c['prompt'],c['seed']


class Qwen21PrunaInputs:
    DESCRIPTION='按当前任务读取主图、选区、保护和有序参考。可选外接端口优先；文件遮罩读取red，主图alpha不是选区。纯文本任务不读取任何图片文件。'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'config':('QWEN21_PRUNA_CONFIG',),
            **{k:('STRING',{'default':'','multiline':k=='reference_filenames','tooltip':'ComfyUI input目录文件名；mask读取red，不是1-alpha。外接端口优先；纯文任务不读取。'}) for k in ('image_filename','mask_filename','protect_filename','reference_filenames')}},
            'optional':{k:(t,{'lazy':True}) for k,t in [('image','IMAGE'),('mask','MASK'),('protect_mask','MASK'),('references','QWEN21_REFS')]},
            'hidden':{'prompt':('PROMPT'),'unique_id':('UNIQUE_ID')}}
    RETURN_TYPES=('IMAGE','MASK','MASK','QWEN21_REFS')
    RETURN_NAMES=('image','selection','protection','references')
    FUNCTION='load'
    @staticmethod
    def active(config):
        if P.is_text(config): return []
        active=['image','protect_mask']
        if P.needs_mask(config):active.append('mask')
        if P.uses_references(config):active.append('references')
        return active
    def check_lazy_status(self,config,**kwargs):
        return [k for k in self.active(config) if k in kwargs and kwargs[k] is None]
    @classmethod
    def IS_CHANGED(cls,config=None,prompt=None,unique_id=None,**kwargs):
        import hashlib
        import folder_paths
        # Fingerprinting runs before linked inputs are evaluated. Inspect literal
        # controls when available, otherwise conservatively hash every file source.
        inputs=(prompt or {}).get(str(unique_id),{}).get('inputs',{})
        link=inputs.get('config')
        if config is None and isinstance(link,list) and len(link)==2:
            source=(prompt or {}).get(str(link[0]),{})
            values=source.get('inputs',{})
            if source.get('class_type')=='SwwanQwen21PrunaControls' and all(isinstance(values.get(k),str) for k in ('task','generation_mode','edit_scope')):
                config=values
        if config is not None and P.is_text(config):return 'text'
        digest=hashlib.sha256()
        files={'image':'image_filename','protect_mask':'protect_filename','mask':'mask_filename','references':'reference_filenames'}
        active=cls.active(config) if config is not None else files
        for socket in active:
            if socket in inputs or socket in kwargs:
                continue
            key=files[socket]
            for name in (kwargs.get(key) if isinstance(kwargs.get(key),str) else '').splitlines():
                if name.strip():
                    p=folder_paths.get_annotated_filepath(name.strip())
                    digest.update((key+'\0'+p+'\0').encode())
                    try:
                        with open(p,'rb') as stream:
                            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                    except FileNotFoundError:
                        # An unused lazy source must not block execution. load()
                        # still reports a missing file when it is actually needed.
                        digest.update(b'missing')
        return digest.hexdigest()
    def load(self,config,image_filename='',mask_filename='',protect_filename='',reference_filenames='',image=None,mask=None,protect_mask=None,references=None,prompt=None,unique_id=None):
        import torch
        if P.is_text(config):return torch.zeros(1,32,32,3),torch.zeros(1,32,32),torch.zeros(1,32,32),[]
        import nodes
        def load_image(name):
            rgb,alpha=nodes.LoadImage().load_image(name)
            if alpha.shape[-2:]==rgb.shape[1:3] and bool(torch.any(alpha)):
                rgb=torch.cat((rgb[...,:3],1-alpha[...,None].to(rgb)),-1)
            return rgb
        if image is None:
            if not image_filename.strip():raise ValueError('请提供主图文件名或连接IMAGE')
            image=load_image(image_filename.strip())
        P.valid_image(image)
        if P.needs_mask(config) and mask is None and mask_filename.strip():mask=getattr(nodes.LoadImageMask(), nodes.LoadImageMask.FUNCTION)(mask_filename.strip(),'red')[0]
        if protect_mask is None and protect_filename.strip():protect_mask=getattr(nodes.LoadImageMask(), nodes.LoadImageMask.FUNCTION)(protect_filename.strip(),'red')[0]
        if not P.needs_mask(config):mask=None
        if not P.uses_references(config):references=[]
        elif references is None:references=[load_image(n.strip()) for n in reference_filenames.splitlines() if n.strip()]
        if len(references)>6:raise ValueError('At most six references')
        if mask is None:mask=torch.zeros_like(image[...,0])
        if protect_mask is None:protect_mask=torch.zeros_like(image[...,0])
        return image,mask,protect_mask,references


class Qwen21PrunaModel:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'config':('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES=('MODEL',)
    FUNCTION='apply'
    DESCRIPTION='在已加载Pruna的干净公共模型上，仅应用当前任务及可选附加LoRA。标准ComfyUI补丁缓存，不宣称GPU常驻或零成本切换。'
    def apply(self,model,config):
        from comfy_execution.graph_utils import GraphBuilder
        graph=GraphBuilder();current=model
        # Match v3 order: Pruna -> optional addon -> task.
        for path,strength in [(config['additional_lora_name'],config['active_additional_strength']),
                              (config['task_lora'],config['active_lora_strength'])]:
            if path and path!='None' and strength:
                current=graph.node('LoraLoaderModelOnly',model=current,lora_name=path,strength_model=strength).out(0)
        if not graph.finalize():return (model,)
        return {'result':(current,),'expand':graph.finalize()}


class Qwen21PrunaPrepare:
    DESCRIPTION='生成干净推理图与同源几何记录：选区扣保护、局部上下文、128..1024动态面积预算、32对齐。扩图自动产生新画布；BFS使用独立人物分支。'
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'config':('QWEN21_PRUNA_CONFIG',)},'optional':{'image':('IMAGE',),'mask':('MASK',),'protect_mask':('MASK',)}}
    RETURN_TYPES=('IMAGE','MASK','QWEN21_PRUNA_SEAM','STRING')
    RETURN_NAMES=('clean_crop','region_mask','geometry','diagnostics')
    FUNCTION='run'
    def run(self,config,image=None,mask=None,protect_mask=None):return P.prepare(image,mask,protect_mask,config)


class Qwen21PrunaMark:
    DESCRIPTION='只在副本标记：普通移除绿色0.4、增强移除绿色1、抠图红色内轮廓。干净裁剪图保留为校色参考，选区不是生成alpha。'
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'mask':('MASK',),'config':('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='run'
    def run(self,image,mask,config):return (P.mark(image,mask,config),)


class Qwen21PrunaRestore:
    DESCRIPTION='GH或BlueBox互斥校色后按唯一几何记录恢复；最后以原坐标mask保护范围外与白保护区。抠图必须是原生RGBA，不用输入mask截断生成主体。'
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'seam':('QWEN21_PRUNA_SEAM',),'config':('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES=('IMAGE','MASK','STRING')
    RETURN_NAMES=('result','composition_mask','diagnostics')
    FUNCTION='run'
    def run(self,image,seam,config):return P.restore(image,seam,config)


class Qwen21PrunaOutput:
    DESCRIPTION='惰性选择BFS或通用结果，不执行未选择采样分支。不要启用未选分支的输出预览，否则该预览会独立触发执行。'
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'config':('QWEN21_PRUNA_CONFIG',)},'optional':{'generic':('IMAGE',{'lazy':True}),'face':('IMAGE',{'lazy':True})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='choose'
    def check_lazy_status(self,config,generic=None,face=None):
        key='face' if config['task']=='face_swap' else 'generic'
        return [key] if (face if key=='face' else generic) is None else []
    def choose(self,config,generic=None,face=None):
        selected=face if config['task']=='face_swap' else generic
        if selected is None:raise ValueError('Selected task output is not connected')
        return (selected,)
