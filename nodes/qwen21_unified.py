# SPDX-License-Identifier: MIT
"""One public control surface and active-only expansion into shared Turbo inference."""
import json
from pathlib import Path
import torch
from ..ops.qwen21_tasks import TASKS, PRESETS, resolve

class Qwen21Controls:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'task':(list(TASKS),{'default':'remove'}),
            'extra_prompt':('STRING',{'default':'','multiline':True}),
            'lora_strength':('FLOAT',{'default':1.,'min':0.,'max':2.,'step':.01}),
            'consistency':(['auto','on','off'],{'default':'auto'}),
            'seed':('INT',{'default':42,'min':0,'max':0xffffffffffffffff}),
            'resolution':('INT',{'default':1024,'min':512,'max':2048,'step':32}),
            'sampling_preset':(list(PRESETS),{'default':'turbo9'}),
            'people':('INT',{'default':1,'min':1,'max':6}),
            'swap_head':('BOOLEAN',{'default':False}),
            'forehead':('BOOLEAN',{'default':False}),
            'bfs_version':(['v1.1','v1.0'],{'default':'v1.1'}),
            'skin_match':('BOOLEAN',{'default':False}),
            'watermark_mode':(['local','full'],{'default':'local'}),
            **{k:('INT',{'default':128,'min':0,'max':4096}) for k in ('left','right','top','bottom')},
            'azimuth':('INT',{'default':0,'min':0,'max':330,'step':30}),
            'elevation':([0,30,60,90],{'default':0}),
            'close_up':('BOOLEAN',{'default':False}),
        }}
    RETURN_TYPES=('QWEN21_CONFIG','STRING')
    RETURN_NAMES=('config','actual_prompt')
    FUNCTION='build'
    def build(self,**kwargs):
        c=resolve(**kwargs)
        return {'ui':{'text':[c['prompt']]},'result':(c,c['prompt'])}


class Qwen21References:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'filenames':('STRING',{'default':'','multiline':True,
            'tooltip':'ComfyUI input filenames, one per line, in person order. Empty = no references.'})},
            'optional':{'config':('QWEN21_CONFIG',),**{f'image{i}':('IMAGE',) for i in range(1,7)}}}
    RETURN_TYPES=('QWEN21_REFS',)
    FUNCTION='load'
    @classmethod
    def IS_CHANGED(cls,filenames='',config=None,**images):
        import hashlib
        import folder_paths
        digest=hashlib.sha256()
        task = (config or {}).get('task')
        if task and task not in ('face_swap','body_swap','outfit_swap','edit','local_edit'): return 'unused'
        for name in filenames.splitlines():
            if name.strip():
                p=Path(folder_paths.get_annotated_filepath(name.strip()))
                if p.exists():
                    with p.open('rb') as stream: digest.update(hashlib.file_digest(stream,'sha256').digest())
        return digest.hexdigest()
    def load(self,filenames='',config=None,**images):
        import nodes
        task = (config or {}).get('task')
        if task and task not in ('face_swap','body_swap','outfit_swap','edit','local_edit'): return ([],)
        names=[x.strip() for x in filenames.splitlines() if x.strip()]
        if task in ('body_swap','outfit_swap'): names=names[:1]
        sockets=[images[k] for k in sorted(images) if images[k] is not None]
        if task in ('body_swap','outfit_swap'): sockets=sockets[:1]
        if names and sockets: raise ValueError('Use either reference filenames or connected images, not both')
        refs=list(sockets)
        if not sockets:
            for name in names:
                rgb,alpha_mask=nodes.LoadImage().load_image(name)
                if alpha_mask.shape[-2:]==rgb.shape[1:3] and bool(torch.any(alpha_mask)):
                    rgb=torch.cat((rgb[...,:3],1-alpha_mask[...,None].to(rgb)),dim=-1)
                refs.append(rgb)
        if len(refs)>6:raise ValueError('At most six reference images')
        return (refs,)


class Qwen21Dispatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'clip':('CLIP',),'vae':('VAE',),
            'image':('IMAGE',),'config':('QWEN21_CONFIG',),'references':('QWEN21_REFS',)},
            'optional':{'mask':('MASK',),'source_alpha_mask':('MASK',)}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('result','report')
    FUNCTION='run'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')
    def run(self,model,clip,vae,image,config,references,mask=None,source_alpha_mask=None):
        from comfy_execution.graph_utils import GraphBuilder
        from ..ops.qwen21_runtime import begin_run
        from ..ops.qwen21_graph import build_face_graph
        config=begin_run(config)
        # Native LoadImage exposes source transparency as an inverted alpha mask.
        # Keep it separate from the edit mask; opaque inputs retain their exact RGB path.
        if source_alpha_mask is not None and bool(torch.any(source_alpha_mask)):
            if source_alpha_mask.shape[-2:] != image.shape[1:3]:
                raise ValueError('Source alpha dimensions must match the image')
            image=torch.cat((image[...,:3],1-source_alpha_mask[:1,...,None].to(image)),dim=-1)
        if mask is None or (mask.shape[-2:]!=image.shape[1:3] and not bool(torch.any(mask))):
            mask=image.new_zeros((1,*image.shape[1:3]))
        if mask.shape[-2:] != image.shape[1:3]:
            raise ValueError('Mask dimensions must match the original image')
        if config['task']=='face_swap':
            return build_face_graph(model,clip,vae,image,mask,references,config)
        graph=GraphBuilder()
        prep=graph.node('SwwanQwen21Prepare',image=image,mask=mask,references=references,config=config)
        core=graph.node('SwwanQwen21Core',model=model,clip=clip,vae=vae,image=prep.out(0),
                        references=prep.out(2),config=config,person=0)
        done=graph.node('SwwanQwen21Finish',image=core.out(0),seam=prep.out(1),config=config,record=core.out(1))
        return {'result':(done.out(0),done.out(2)),'expand':graph.finalize()}


class Qwen21Prepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image':('IMAGE',),'mask':('MASK',),'references':('QWEN21_REFS',),'config':('QWEN21_CONFIG',)}}
    RETURN_TYPES=('IMAGE','SEAM','QWEN21_REFS')
    FUNCTION='run'
    def run(self,image,mask,references,config):
        from ..ops.qwen21_runtime import prepare
        return prepare(image,mask,references,config)


class Qwen21Core:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'clip':('CLIP',),'vae':('VAE',),'image':('IMAGE',),
            'references':('QWEN21_REFS',),'config':('QWEN21_CONFIG',),'person':('INT',{'default':0})},
            'optional':{'previous':('IMAGE',)}}
    RETURN_TYPES=('IMAGE','QWEN21_RECORD')
    FUNCTION='run'
    def run(self,model,clip,vae,image,references,config,person=0,previous=None):
        from ..ops.qwen21_runtime import infer
        return infer(model,clip,vae,image,references,config,person,previous)


class Qwen21Finish:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'image':('IMAGE',),'seam':('SEAM',),'config':('QWEN21_CONFIG',),'record':('QWEN21_RECORD',)}}
    RETURN_TYPES=('IMAGE','MASK','STRING')
    FUNCTION='run'
    def run(self,image,seam,config,record):
        from ..ops.qwen21_runtime import finish
        return finish(image,seam,config,record)


class Qwen21Source:
    @classmethod
    def INPUT_TYPES(cls): return {'required':{'image':('IMAGE',),'mask':('MASK',)}}
    RETURN_TYPES=('IMAGE','MASK')
    FUNCTION='run'
    def run(self,image,mask):return image,mask


class Qwen21ReferenceAtIndex:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'references':('QWEN21_REFS',),'index':('INT',{'default':0})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='run'
    def run(self,references,index):
        if index<0 or index>=len(references):raise ValueError(f'Missing reference for person {index+1}')
        return (references[index],)


class Qwen21Auxiliary:
    @classmethod
    def INPUT_TYPES(cls): return {'required':{'image':('IMAGE',)}}
    RETURN_TYPES=('QWEN21_REFS',)
    FUNCTION='run'
    def run(self,image):
        from ..ops.qwen21_runtime import auxiliary
        return ([auxiliary(image)],)


class Qwen21FaceGuard:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'count':('INT',{'default':1}),
        'valid':('BOOLEAN',{'default':False}),'references':('QWEN21_REFS',),'config':('QWEN21_CONFIG',)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='run'
    def run(self,image,count,valid,references,config):
        from ..ops.qwen21_runtime import metadata
        if not valid or count<1: raise ValueError('No valid face/head regions; no sampling started')
        if len(references)<count: raise ValueError(f'{count} people require {count} references; received {len(references)}')
        import time
        metadata(config,{'status':'regions_ready','count':count},'face-progress.json')
        metadata(config,{'detection_s':time.time()-config['started_at'],'count':count},'face-stages.json')
        return (image,)


class Qwen21FaceCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'person':('INT',{'default':0}),'config':('QWEN21_CONFIG',)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='run'
    def run(self,image,person,config):
        from ..ops.qwen21_runtime import metadata,save_tensor
        import time
        progress={'status':'processing','person':person+1,'started_at':time.time()}
        metadata(config,progress,'face-progress.json')
        metadata(config,progress,f'person-{person+1:02d}-start.json')
        save_tensor(Path(config['artifact_dir'])/f'person-{person+1:02d}-before.png',image)
        return (image,)


class Qwen21FaceRestore:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'background':('IMAGE',),'image':('IMAGE',),'seam':('SEAM',),
        'mask':('MASK',),'manual':('BOOLEAN',{'default':False}),'config':('QWEN21_CONFIG',),
        'record':('QWEN21_RECORD',),'person':('INT',{'default':0})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='run'
    def run(self,background,image,seam,mask,manual,config,record,person):
        from ..ops.qwen21_geometry import restore_aligned
        from ..ops.qwen21_runtime import save_tensor,metadata,flush_saves
        import time
        start=time.perf_counter()
        x0,y0,x1,y1=seam['box']; radius=round(min(x1-x0,y1-y0)*.05)
        result,fusion=restore_aligned(background,image,seam,cropped_mask=mask,
            expand=radius,feather=radius,strict=manual)
        record=dict(record,paste_back_s=time.perf_counter()-start,status='person_completed',manual_mask=manual)
        record['protected_changed_values']=int(torch.count_nonzero((result-background)*(fusion==0)[...,None]))
        path=Path(config['artifact_dir']);tag=f'person-{person+1:02d}'
        save_tensor(path/(tag+'-final.png'),result);save_tensor(path/(tag+'-fusion-mask.png'),fusion[...,None])
        flush_saves(config)
        metadata(config,record,tag+'.json')
        metadata(config,{'status':'person_completed','person':person+1},'face-progress.json')
        return (result,)


class Qwen21FaceFinish:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'image':('IMAGE',),'preview':('IMAGE',),'config':('QWEN21_CONFIG',)}}
    RETURN_TYPES=('IMAGE','STRING')
    FUNCTION='run'
    def run(self,image,preview,config):
        from ..ops.qwen21_runtime import save_tensor,metadata
        path=Path(config['artifact_dir'])
        records=[json.loads(p.read_text()) for p in sorted(path.glob('person-*.json')) if not p.name.endswith('-start.json')]
        if not records or any(x.get('status')!='person_completed' for x in records):raise RuntimeError('Incomplete face task')
        report=dict(config,status='completed',people_processed=len(records),people_records=records)
        save_tensor(path/'final.png',image);save_tensor(path/'person-order.png',preview)
        metadata(config,report)
        return image,json.dumps(report,ensure_ascii=False)


class Qwen21FaceSequence:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'clip':('CLIP',),'vae':('VAE',),
        'image':('IMAGE',),'mask':('MASK',),'regions':('SEGS',),'manual':('BOOLEAN',{'default':False}),
        'group_threshold':('FLOAT',{'default':50.}),'count':('INT',{'default':1}),
        'preview':('IMAGE',),'references':('QWEN21_REFS',),'config':('QWEN21_CONFIG',)}}
    RETURN_TYPES=('IMAGE','STRING')
    FUNCTION='run'
    def run(self,model,clip,vae,image,mask,regions,manual,group_threshold,count,preview,references,config):
        from ..ops.qwen21_graph import build_face_sequence
        return build_face_sequence(model,clip,vae,image,mask,regions,manual,group_threshold,count,preview,references,config)


class Qwen21FaceRegions:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'segs':('SEGS',),'index':('INT',{'default':-1,'min':-1,'max':5})}}
    RETURN_TYPES=('SEGS','MASK')
    FUNCTION='run'
    def run(self,segs,index=-1):
        from ..ops.qwen21_graph import row_order
        from ..ops.segment_order import SegmentOrderAlgorithm
        ordered=row_order(segs) if index<0 else segs
        shape,items=ordered
        if index>=0:
            if index>=len(items):raise ValueError(f'Missing frozen region {index+1}')
            items=[items[index]]
        mask=SegmentOrderAlgorithm().create_full_mask(shape,items,[s.crop_region for s in items])
        return (shape,items),mask.unsqueeze(0)
