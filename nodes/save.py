# SPDX-License-Identifier: GPL-3.0-only
"""Unified saver: straight / premultiplied images, paths and encoder controls."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageColor
import folder_paths
from ..ops.image_save import _build_save_kwargs, _save_image_with_fallback, _build_png_metadata
from ..rgba_nodes import _resize_alpha, _unpremultiply, _flatten_premultiplied


class SwwanSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',), 'file_format': (['png','jpg','webp','tif','bmp'], {'default':'png'}),
            'output_path': ('STRING', {'default':''}), 'filename_prefix': ('STRING', {'default':'Swwan'}),
        }, 'optional': {
            'alpha': ('MASK',), 'input_mode': (['straight','premultiplied'], {'default':'straight'}),
            'alpha_mode': (['auto','keep','flatten'], {'default':'auto'}),
            'has_alpha': ('BOOLEAN', {'default':True}), 'background_color': ('COLORCODE', {'default':'#ffffff'}),
            'quality': ('INT', {'default':88,'min':1,'max':100}),
            'png_compress_level': ('INT', {'default':1,'min':0,'max':9}),
            'optimize': ('BOOLEAN', {'default':True}), 'webp_lossless': ('BOOLEAN', {'default':False}),
            'webp_method': ('INT', {'default':2,'min':0,'max':6}),
            'number_prefix': ('BOOLEAN', {'default':False}), 'number_digits': ('INT', {'default':5,'min':1,'max':10}),
            'embed_metadata': ('BOOLEAN', {'default':True}), 'save_workflow_as_json': ('BOOLEAN', {'default':False}),
            'caption': ('STRING', {'default':'','multiline':True}),
            'epsilon': ('FLOAT', {'default':0.001,'min':0.000001,'max':0.1}),
        }, 'hidden': {'prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('paths',)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = 'save'
    CATEGORY = 'Swwan/IO'
    DESCRIPTION = 'Relative folders use ComfyUI output. Straight RGB/RGBA or explicitly premultiplied RGB + alpha. JPEG/BMP flatten alpha. Absolute paths are returned for every image.'

    def save(self, image, file_format, output_path='', filename_prefix='Swwan', alpha=None,
             input_mode='straight', alpha_mode='auto', has_alpha=True, background_color='#ffffff',
             quality=88, png_compress_level=1, optimize=True, webp_lossless=False, webp_method=2,
             number_prefix=False, number_digits=5, embed_metadata=True, save_workflow_as_json=False,
             caption='', epsilon=0.001, prompt=None, extra_pnginfo=None):
        if file_format not in {'png','jpg','webp','tif','bmp'}:
            raise ValueError('Unsupported image format: '+str(file_format))
        if image.shape[0] == 0:
            raise ValueError('Save Image requires at least one image.')
        root = Path(folder_paths.get_output_directory()).resolve()
        if isinstance(output_path, (list,tuple)):
            if len(output_path)!=image.shape[0]:raise ValueError('Output folder count must match image count.')
            folders=list(output_path)
        else:folders=[output_path]*image.shape[0]
        # Use the ComfyUI resolver for prefix expansion and subfolder handling.
        paths, previews = [], []
        for i, folder in enumerate(folders):
            base=Path(folder) if folder and Path(folder).is_absolute() else root / (folder or '')
            base=base.resolve();base.mkdir(parents=True,exist_ok=True)
            resolved, stem, _, _, _=folder_paths.get_save_image_path(filename_prefix,str(base),image.shape[2],image.shape[1])
            directory=Path(resolved);directory.mkdir(parents=True,exist_ok=True)
            rgb=image[i:i+1,...,:3].float()
            if rgb.shape[-1]==1:rgb=rgb.expand(-1,-1,-1,3)
            a=alpha if alpha is not None else image[...,3] if image.shape[-1]==4 else None
            if a is not None:a=_resize_alpha(a.to(image.device).float(),image.shape[0],image.shape[1],image.shape[2])[i:i+1]
            keep=(a is not None and bool(has_alpha) and alpha_mode!='flatten' and file_format in {'png','webp','tif'})
            if a is not None:
                color=torch.tensor(ImageColor.getrgb(background_color),device=rgb.device,dtype=rgb.dtype)/255
                if input_mode=='premultiplied':
                    rgb=_unpremultiply(rgb,a,epsilon) if keep else _flatten_premultiplied(rgb,a,color)
                elif not keep:rgb=rgb*a.unsqueeze(-1)+color*(1-a.unsqueeze(-1))
            elif input_mode=='premultiplied' and has_alpha:
                raise ValueError('Premultiplied input requires an alpha mask.')
            data=np.clip(rgb[0].cpu().numpy()*255,0,255).astype(np.uint8)
            if keep:data=np.dstack((data,np.clip(a[0].cpu().numpy()*255,0,255).astype(np.uint8)))
            pil=Image.fromarray(data)
            kwargs=_build_save_kwargs(file_format,quality,png_compress_level,optimize,webp_lossless,webp_method)
            if file_format=='webp':kwargs['exact']=True
            if file_format=='png' and embed_metadata:kwargs['pnginfo']=_build_png_metadata(prompt,extra_pnginfo)
            counter=0
            while True:
                number=f'{counter:0{number_digits}d}'
                filename=f'{number}_{stem}' if number_prefix else f'{stem}_{number}'
                path=directory/(filename+'.'+file_format)
                if any((directory/(filename+'.'+ext)).exists() for ext in ('png','jpg','webp','tif','bmp','txt','json')):
                    counter+=1;continue
                # Reserve a name before encoding; concurrent batches never overwrite.
                try:
                    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                except FileExistsError:
                    counter+=1;continue
                try:
                    with os.fdopen(fd,'wb') as handle:_save_image_with_fallback(pil,handle,kwargs)
                except Exception:
                    path.unlink(missing_ok=True);raise
                break
            paths.append(str(path.resolve()))
            if caption:
                with path.with_suffix('.txt').open('x',encoding='utf-8') as handle:handle.write(caption)
            if save_workflow_as_json and (extra_pnginfo or {}).get('workflow') is not None:
                with path.with_suffix('.json').open('x',encoding='utf-8') as handle:json.dump(extra_pnginfo['workflow'],handle,ensure_ascii=False)
            try:relative=path.relative_to(root)
            except ValueError:continue
            previews.append({'filename':relative.name,'subfolder':str(relative.parent) if relative.parent!=Path('.') else '', 'type':'output'})
        return {'result':(paths,), 'ui':{'images':previews}}
