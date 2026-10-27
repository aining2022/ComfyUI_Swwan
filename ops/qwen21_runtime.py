# SPDX-License-Identifier: MIT
"""Per-request clean patches, bounded CPU LoRA cache and serialized Turbo sampling."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import torch
from .qwen21_geometry import prepare_aligned, resize_image, restore_aligned

GPU_LOCK=threading.RLock()
SAVE_POOL=ThreadPoolExecutor(max_workers=2,thread_name_prefix='qwen21-save')
PENDING_SAVES={}

class LoraCache:
    def __init__(self,limit=2*1024**3):
        self.limit=limit; self.bytes=0; self.items=OrderedDict(); self.lock=threading.RLock()

    def get(self,path,loader):
        path=Path(path); stat=path.stat()
        key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        with self.lock:
            if key in self.items:
                value,size=self.items.pop(key); self.items[key]=(value,size)
                return value,True,key
            value=loader(str(path))
            size=sum(t.numel()*t.element_size() for t in value.values() if torch.is_tensor(t))
            while self.items and self.bytes+size>self.limit:
                _,(_,old)=self.items.popitem(last=False); self.bytes-=old
            if size<=self.limit:
                self.items[key]=(value,size); self.bytes+=size
            return value,False,key

LORA_CACHE=LoraCache()


def image_digest(image):
    a=image.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_tensor(path,image):
    from PIL import Image
    a=(image[0].detach().float().cpu().clamp(0,1).numpy()*255).round().astype('uint8')
    if a.shape[-1]==1: a=a[...,0]
    Image.fromarray(a).save(path)


def begin_run(config):
    import folder_paths
    c=dict(config); c['run_id']=uuid.uuid4().hex
    path=Path(folder_paths.get_output_directory())/'qwen21-unified'/c['run_id']
    path.mkdir(parents=True,exist_ok=False); c['artifact_dir']=str(path)
    c['started_at']=time.time()
    (path/'request.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
    (path/'prompt.txt').write_text(c['prompt'])
    return c


def metadata(config,record,name='result.json'):
    Path(config['artifact_dir'],name).write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str))


def flush_saves(config):
    for future in PENDING_SAVES.pop(config["run_id"], []): future.result()


def auxiliary(image,budget=768):
    # Preserve field of view and aspect ratio. Padding only to the next 32-pixel cell.
    import math
    import torch.nn.functional as F
    h,w=image.shape[1:3]; scale=budget/math.sqrt(h*w)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    x=resize_image(image[:1],nw,nh)
    pw,ph=(-nw)%32,(-nh)%32
    x=F.pad(x.movedim(-1,1),(pw//2,pw-pw//2,ph//2,ph-ph//2),mode='replicate').movedim(1,-1)
    return x


def prepare(image,mask,refs,config):
    started=time.perf_counter()
    task=config['task']; source=image[:1]
    if task in ('body_swap','outfit_swap') and len(refs)!=1: raise ValueError(task+' requires exactly one reference image')
    if len(refs)>2 and task!='face_swap': raise ValueError('At most two auxiliary reference images per generic task')
    if task=='outpaint':
        l,r,t,b=[config[k] for k in ('left','right','top','bottom')]
        h,w=source.shape[1:3]
        expanded=source.new_full((1,h+t+b,w+l+r,source.shape[-1]),128/255)
        if source.shape[-1]==4: expanded[...,3]=1
        expanded[:,t:t+h,l:l+w]=source
        selected=source.new_ones((1,h+t+b,w+l+r));selected[:,t:t+h,l:l+w]=0
        prepared,_,seam=prepare_aligned(expanded,None,config['resolution'],full_image=True)
        seam['original_mask']=selected;seam['original_image']=expanded
    else:
        full=task in ('edit','enhance','angles','body_swap','outfit_swap') or (task=='watermark' and config['watermark_mode']=='full')
        prepared,selected,seam=prepare_aligned(source,mask,config['resolution'],full_image=full)
        if task=='remove':
            color=prepared.new_tensor([0,1,0]);prepared=prepared.clone()
            prepared[...,:3]=prepared[...,:3]*(1-selected[...,None]*.4)+color*selected[...,None]*.4
        elif task=='cutout':
            import torch.nn.functional as F
            support=(selected>0).float()[:,None]
            border=(F.max_pool2d(support,5,1,2)+F.max_pool2d(-support,5,1,2))[:,0].clamp(0,1)
            prepared=prepared.clone()
            prepared[...,:3]=prepared[...,:3]*(1-border[...,None])+prepared.new_tensor([1,0,0])*border[...,None]
    refs=[auxiliary(x) for x in refs]
    if config.get('artifact_dir'):
        metadata(config,dict(prepare_s=time.perf_counter()-started,canvas_size=seam['canvas_size'],
                            box=seam['box'],virtual_box=seam['virtual_box'],content_box=seam['content_box'],
                            source_padding=seam['source_padding'],scale=seam['scale']),'geometry.json')
    return prepared,seam,refs


def encode(clip,vae,prompt,images):
    """Core 2.1 resolution=0 semantics; CFG=1 encodes only the positive condition."""
    import node_helpers
    import comfy.model_management
    latents=[]; vision=[]
    for image in images:
        h,w=image.shape[1:3]
        if h%32 or w%32: raise ValueError('Prepared references must already be aligned to 32')
        image=image[:1]
        rgb=image[...,:3]
        if image.shape[-1]==4: rgb=rgb*image[...,3:4]+1-image[...,3:4]
        vision.append(rgb)
        latents.append(vae.encode(image))
    tokens=clip.tokenize(prompt,images=vision,keep_vision=False,prevent_empty_text=True)
    cond=clip.encode_from_tokens_scheduled(tokens)
    cond=node_helpers.conditioning_set_values(cond,{'reference_latents':latents},append=True)
    # Latent shape is taken from the actual encoded main reference, never a second size control.
    latent={'samples':torch.zeros_like(latents[0],device=comfy.model_management.intermediate_device())}
    return cond,latent


def infer(base,clip,vae,image,refs,config,person=0,previous=None):
    import folder_paths
    import comfy.sd
    import comfy.utils
    import comfy.samplers
    import comfy.sample
    import latent_preview
    import nodes
    record=dict(config);record['person']=int(person)+1 if config['task']=='face_swap' else None
    tag=f'person-{person+1:02d}' if config['task']=='face_swap' else 'generation'
    path=Path(config['artifact_dir']); futures=[]; started=time.perf_counter()
    phase_file=path/(f'person-{person+1:02d}-start.json' if config['task']=='face_swap' else 'geometry.json')
    if phase_file.exists():
        phase=json.loads(phase_file.read_text())
        record['prepare_s']=time.time()-phase['started_at'] if 'started_at' in phase else phase.get('prepare_s',0)
    stage_file=path/'face-stages.json'
    if stage_file.exists(): record['detection_s']=json.loads(stage_file.read_text()).get('detection_s',0)
    images=[image]+refs
    record['reference_shapes']=[list(x.shape[1:3]) for x in images]
    record['reference_sha256']=[image_digest(x) for x in images]
    fingerprint={k:record[k] for k in ('task','lora','lora_strength','prompt','reference_shapes','reference_sha256','sigmas','seed')}
    record['cache_identity']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    futures.append(SAVE_POOL.submit(save_tensor,path/(tag+'-input.png'),image.detach().cpu().clone()))
    if previous is not None: save_tensor(path/(tag+'-before.png'),previous)
    try:
        wait=time.perf_counter()
        with GPU_LOCK:
            record['gpu_queue_s']=time.perf_counter()-wait
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter(); model=base.clone()
            # Every request starts from the loader's clean model, including person 2..6.
            if config['lora']:
                filename=folder_paths.get_full_path_or_raise('loras',config['lora'])
                weights,hit,key=LORA_CACHE.get(filename,lambda p:comfy.utils.load_torch_file(p,safe_load=True))
                model,_=comfy.sd.load_lora_for_models(model,None,weights,config['lora_strength'],0)
                record.update(lora_cache_hit=hit,lora_file=key,lora_cache_bytes=LORA_CACHE.bytes)
            else: record.update(lora_cache_hit=False,lora_cache_bytes=LORA_CACHE.bytes)
            fingerprint['lora_file']=record.get('lora_file')
            record['cache_identity']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            sage=nodes.NODE_CLASS_MAPPINGS.get('QwenImage21SageAttentionT8')
            if sage is None: raise RuntimeError('QwenImage21SageAttentionT8 is required')
            model=sage.execute(model,backend_mode='sage_kitchen')[0]
            record['lora_patch_s']=time.perf_counter()-t
            t=time.perf_counter(); positive,latent=encode(clip,vae,config['prompt'],images)
            record['encode_s']=time.perf_counter()-t
            noise=comfy.sample.prepare_noise(latent['samples'],config['seed'])
            sigma=torch.tensor(config['sigmas'],dtype=torch.float32)
            callback=latent_preview.prepare_callback(model,len(sigma)-1,{})
            t=time.perf_counter()
            samples=comfy.sample.sample_custom(model,noise,1.,comfy.samplers.ksampler('euler'),sigma,
                positive,positive,latent['samples'],callback=callback,seed=config['seed'])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            record['sample_s']=time.perf_counter()-t
            t=time.perf_counter(); result=vae.decode(samples)
            record['decode_s']=time.perf_counter()-t
            record['peak_vram_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if result.shape[1:3]!=image.shape[1:3]: raise RuntimeError('Decoded size differs from encoded main reference')
        futures.append(SAVE_POOL.submit(save_tensor,path/(tag+'-raw.png'),result.detach().cpu().clone()))
        record['status']='sampled';record['total_core_s']=time.perf_counter()-started
        PENDING_SAVES.setdefault(config['run_id'],[]).extend(futures)
        metadata(config,record,tag+'.json')
        return result,record
    except Exception as error:
        record.update(status='failed',error=repr(error),total_core_s=time.perf_counter()-started)
        metadata(config,record,tag+'.json');metadata(config,record,'failure.json')
        raise RuntimeError(f"Qwen {tag} failed; diagnostic artifacts: {path}: {error}") from error


def finish(image,seam,config,record):
    t=time.perf_counter();task=config['task']
    if task in ('enhance','angles'): result=image[...,:3]; mask=torch.ones_like(result[...,0])
    else:
        result,mask=restore_aligned(seam['original_image'],image,seam,transparent=task=='cutout')
    record.update(paste_back_s=time.perf_counter()-t,status='completed',output_size=list(result.shape[1:3]))
    save_tensor(Path(config['artifact_dir'])/'final.png',result)
    save_tensor(Path(config['artifact_dir'])/'fusion-mask.png',mask[...,None])
    if task not in ('enhance','angles','cutout'):
        record['protected_changed_values']=int(torch.count_nonzero((result-seam['original_image'])*(mask==0)[...,None]))
    flush_saves(config)
    metadata(config,record)
    return result,mask,json.dumps(record,ensure_ascii=False)
