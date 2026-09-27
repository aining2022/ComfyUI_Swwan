"""Portable CPU parity and workflow migration tests for pure face/head preprocessing.

No model downloads, detector imports, sibling plugins or Downloads files are needed.
"""
import argparse
import ast
import copy
import json
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from migrate_qwen2511_workflow import load_nodes, defaults
from migrate_qwen_face_head_workflow import migrate, MAPPINGS
p = argparse.ArgumentParser(); p.add_argument('--comfyui-root', type=Path, required=True)
args, rest = p.parse_known_args(); registry = load_nodes(args.comfyui_root)
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import cv2
import kornia
import torchvision.transforms as T
import comfy.utils
import comfy.model_management
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from typing import List, Tuple, Dict, Union
from nodes import MAX_RESOLUTION
from importlib import import_module
common = import_module(registry.__name__ + '.layerstyle_utils')
refs = dict(torch=torch, np=np, scipy=scipy, cv2=cv2, F=F, T=T, kornia=kornia,
            comfy=comfy, Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont,
            ImageFilter=ImageFilter, ImageOps=ImageOps, os=__import__('os'),
            sys=sys, List=List, Tuple=Tuple, Dict=Dict, Union=Union, MAX_RESOLUTION=MAX_RESOLUTION,
            __file__=str(ROOT / 'nodes' / 'reference.py'), main_device=torch.device('cpu'),
            tqdm=lambda x, **kwargs: x, tensor2pil=common.tensor2pil, pil2tensor=common.pil2tensor)
FIXTURES = ROOT / 'tests/fixtures/face_processing'
exec((FIXTURES/'kj_conversion.py').read_text(), refs)
for path in sorted(FIXTURES.glob('*Algorithm.py')):
    tree = ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), str(path), 'exec'), refs)
# Standalone original Impact functions use their original helper namespace and SEG type.
impact = dict(refs, logging=__import__('logging'), SEG=namedtuple('SEG', ['cropped_image','cropped_mask','confidence','crop_region','bbox','label','control_net_wrapper'], defaults=[None]))
exec((FIXTURES/'impact_utils.py').read_text(), impact)
impact['utils'] = types.SimpleNamespace(make_crop_region=impact['make_crop_region'])
exec((FIXTURES/'impact_core.py').read_text(), impact)
layer = dict(refs)
exec((FIXTURES/'layer_grow.py').read_text(), layer)
was = dict(refs, pil2mask=lambda image: 1. - torch.from_numpy(np.array(image.convert('L')).astype(np.float32)/255.))
exec((FIXTURES/'was_fill.py').read_text(), was)
was['WAS_Tools_Class'] = lambda: types.SimpleNamespace(Masking=types.SimpleNamespace(fill_region=was['fill_region']))


def run(node_id, **kwargs):
    cls=registry.NODE_CLASS_MAPPINGS[node_id]; values=defaults(cls); values.update(kwargs)
    if 'unique_id' in cls.INPUT_TYPES().get('hidden', {}): values['unique_id']=None
    return getattr(cls(), cls.FUNCTION)(**values)


def same(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    else: assert a==b,(a,b)


class FaceHeadProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(173)
        cls.image=torch.randint(0,256,(2,48,64,3)).float()/255
        cls.mask=torch.zeros(2,48,64); cls.mask[:,9:36,10:49]=1; cls.mask[:,18:23,21:27]=0
        cls.other=torch.zeros(1,48,64); cls.other[:,20:42,36:57]=.7
        cls.original=json.loads((ROOT/'tests/fixtures/qwen-face-head-original.json').read_text())

    def test_workflow_preservation_and_idempotence(self):
        out=migrate(self.original,registry); same(out,migrate(out,registry))
        same(out,json.loads((ROOT/'examples/qwen-face-head-swwan.json').read_text()))
        old={n['id']:n for n in self.original['nodes']}; nodes={n['id']:n for n in out['nodes']}
        self.assertEqual(nodes.keys(),old.keys()); self.assertEqual(len(out['links']),len(self.original['links'])+2)
        for node_id,node in nodes.items():
            before=old[node_id]; self.assertEqual(node['pos'],before['pos']);self.assertEqual(node['size'],before['size'])
            self.assertEqual(node['mode'],before['mode'])
            if before['type'] not in MAPPINGS:
                candidate=copy.deepcopy(node)
                # forLoopStart receives only the necessary original-image connection for restore.
                candidate['outputs']=before.get('outputs',[])
                self.assertEqual(candidate,before)
                continue
            cls=registry.NODE_CLASS_MAPPINGS[node['type']]
            self.assertEqual(len(node['widgets_values']),len(defaults(cls)))
            params=dict(zip(defaults(cls),node['widgets_values']))
            schema={**cls.INPUT_TYPES().get('required',{}),**cls.INPUT_TYPES().get('optional',{})}
            for name,value in params.items():
                if isinstance(schema[name][0],list):self.assertIn(value,schema[name][0],(node_id,name))
            for inp in node['inputs']:
                self.assertIn(inp['name'],schema,(node_id,inp))
                if inp.get('widget'):self.assertEqual(inp['widget']['name'],inp['name'])
        for link_id,source,slot,target,index,kind in out['links']:
            output=nodes[source]['outputs'][slot]; inp=nodes[target]['inputs'][index]
            self.assertIn(link_id,output.get('links') or []); self.assertEqual(inp['link'],link_id)
            self.assertTrue(kind=='*' or output['type'] in (kind,'*'),(link_id,output,kind))
            self.assertTrue(kind=='*' or inp['type'] in (kind,'*'),(link_id,inp,kind))
        self.assertEqual(nodes[199],old[199]) # Explicitly ignored LG node.
        self.assertEqual(nodes[131]['inputs'][next(l[4] for l in out['links'] if l[0]==147)]['name'],'edit_top_factor')
        self.assertEqual(nodes[175]['inputs'][next(l[4] for l in out['links'] if l[0]==194)]['name'],'edge_length')
        self.assertEqual(nodes[70]['inputs'][next(l[4] for l in out['links'] if l[0]==71)]['name'],'mask_expand')

    def test_mask_process_reference(self):
        for level in [0.,.002,.02,1.]:
            mask=self.mask[:1]*level
            expect=was['WAS_Mask_Fill_Region']().fill_region(mask)[0]
            same(run('SwwanMaskProcess',mask=mask,operation='fill_holes')[0],expect)
        for threshold in [1,5,20,255]:
            for dtype in (torch.float32,torch.float64):
                mask=(self.mask*.8).to(dtype);expect=mask.clone().cpu();expect[expect>threshold/255.]=1;expect[expect<=threshold/255.]=0
                same(run('SwwanMaskProcess',mask=mask,operation='binary',threshold=threshold)[0],expect)
        for options in [dict(erode_dilate=3,fill_holes=4,remove_isolated_pixels=2,smooth=3,blur=5),dict(erode_dilate=-3,fill_holes=0,remove_isolated_pixels=0,smooth=0,blur=0)]:
            expect=refs['MaskFix']().execute(self.mask,**options)[0]
            values=dict(options);values['close_holes']=values.pop('fill_holes')
            same(run('SwwanMaskProcess',mask=self.mask,operation='cleanup',**values)[0],expect)
        for grow in [-3,0,4]:
            for invert in [True,False]:
                expected=[]
                for m in self.mask:
                    if invert:m=1-m
                    pil=layer['tensor2pil'](m.unsqueeze(0)).convert('L')
                    expected.append(layer['expand_mask'](layer['image2mask'](pil),grow,2))
                same(run('SwwanMaskProcess',mask=self.mask,operation='layer_grow',grow=grow,blur=2,invert_mask=invert)[0],torch.cat(expected))
        for expand,blur in [(-2,0),(0,0),(3,1.5)]:
            options=dict(expand=expand,tapered_corners=True,flip_input=False,blur_radius=blur,incremental_expandrate=.3,lerp_alpha=.7,decay_factor=.8,fill_holes=True)
            same(run('SwwanMaskProcess',mask=self.mask,operation='grow_blur',**options),refs['GrowMaskWithBlur']().expand_mask(self.mask,**options))
        out=run('SwwanMaskProcess',mask=self.mask,operation='fill_holes')[0]
        self.assertEqual(out.shape,(2,1,48,64)) # Deliberate fix for original WAS's ambiguous batch path.
        self.assertEqual(run('SwwanMaskProcess',mask=out,operation='binary')[0].shape,self.mask.shape)

    def test_mask_combine_analysis_reference(self):
        cls=refs['MaskBlendOperation'];schema=cls.INPUT_TYPES()['required']
        for op in schema['混合模式'][0]:
            for bbox in schema['BBOX'][0]:
                for align in schema['对齐方式'][0]:
                    same(run('SwwanMaskCombine',operation=op,bbox_mode=bbox,alignment=align,mask_1=self.mask[:1],mask_2=self.other),cls().execute(op,bbox,align,self.mask[:1],self.other))
        same(run('SwwanMaskCombine',operation='相加',bbox_mode='关闭',alignment='左对齐'),cls().execute('相加','关闭','左对齐'))
        for mask in [self.mask,torch.zeros_like(self.mask),self.mask*.5]:
            for options in [(0,0,0,0),(25,10,-20,-70),(-90,-90,-90,-90)]:
                out=run('SwwanMaskAnalyze',mask=mask,**dict(zip(['top_percent','bottom_percent','left_percent','right_percent'],options)),minimum_area_percent=.1)
                same(out[:7],refs['孤海遮罩分析']().analyze_mask(mask,*options))
                same(out[7:],refs['GuHaiMaskDetect']().detect(mask,.1))

    def test_segments_reference(self):
        for combined in [True,False]:
            for contour in [True,False]:
                for bbox in [True,False]:
                    expected=impact['mask_to_segs'](self.mask[:1].squeeze(0),combined,3.,bbox,10,is_contour=contour)
                    out=run('SwwanMaskSegments',operation='from_mask',mask=self.mask[:1],combined=combined,bbox_fill=bbox,contour_fill=contour)
                    same(out[0],expected);same(out[1],impact['segs_to_combined_mask'](expected).unsqueeze(0))
        masks=torch.zeros(1,48,64);masks[:,2:18,2:18]=.73;masks[:,20:46,35:62]=.92
        segs=impact['mask_to_segs'](masks.squeeze(0),False,3.,False,1,is_contour=False)
        same(run('SwwanMaskSegments',operation='mask_batch',segs=segs)[1],torch.stack(impact['segs_to_masklist'](segs)))
        for rule in refs['孤海Seg次序过滤'].INPUT_TYPES()['required']['优先规则'][0]:
            for order in ['正序','反序']:
                for start,count in [(0,1),(1,1),(8,2),(1,0),(8,0)]:
                    same(run('SwwanMaskSegments',operation='filter',segs=segs,sort_rule=rule,sort_order=order,start_index=start,count=count,group_threshold=20),refs['孤海Seg次序过滤']().filter_segments(segs,rule,order,start,count,20))
        empty=((48,64),[])
        same(run('SwwanMaskSegments',operation='mask_batch',segs=empty)[1],torch.stack(impact['segs_to_masklist'](empty)))

    def test_matte_and_preview_reference(self):
        for crop in [False,True]:
            for stroke in [0,2]:
                same(run('SwwanImageMatte',image=self.image,mask=self.mask,fill_holes=True,crop_mask=crop,crop_factor=1.2,stroke_width=stroke,background_color='#ffffff'),refs['RemoveBackgroundWithMask']().remove_background(self.image,self.mask,True,crop,1.2,stroke,'#ffffff'))
        for show in [False,True]:
            same(run('SwwanImageAndMaskPreview',image=self.image,mask=self.mask,preview_mode='regions',mask_opacity=.4,region_color='#ff00a2',show_numbers=show,number_opacity=.8,number_scale='跟随遮罩缩放',font_scale=.5,number_font='FreeMono.ttf',number_color='#ffffff'),refs['ImageMaskPreview_Guhai']().preview(self.image,self.mask,.4,'#ff00a2',show,'FreeMono.ttf',.8,'跟随遮罩缩放',.5,'#ffffff'))

    def test_resize_color_reference(self):
        for method in refs['ImageResize'].INPUT_TYPES()['required']['method'][0]:
            for condition in refs['ImageResize'].INPUT_TYPES()['required']['condition'][0]:
                for interpolation in ['lanczos','nearest','bicubic']:
                    same(run('ImageResizeKJv2Alternative',image=self.image,width=36,height=30,divisible_by=8,resize_mode='essentials',essentials_method=method,essentials_condition=condition,essentials_interpolation=interpolation)[:3],refs['ImageResize']().execute(self.image,36,30,method,interpolation,condition,8))
        for space in refs['ImageColorMatch'].INPUT_TYPES()['required']['color_space'][0]:
            for mask in [None,self.mask[:1]]:
                expected=refs['ImageColorMatch']().execute(self.image,self.image[:1],space,.7,'cpu',1,mask)
                same(run('SwwanColorMatch',image_target=self.image,image_ref=self.image[:1],match_mode='mean_std',color_space=space,factor=.7,device='cpu',batch_size=1,reference_mask=mask),expected)

    def test_simulated_chain(self):
        crop=run('SwwanCropByMaskV5',image=self.image[:1],mask=self.mask[:1],crop_mode='edit_region',edit_expansion_mode='factor',edit_top_factor=1.2,edit_bottom_factor=3.,edit_left_factor=1.2,edit_right_factor=1.2,alignment=0)
        reference=import_module(registry.__name__+'.edit_region').EditRegionCrop().裁剪图像(self.image[:1],self.mask[:1],False,1.2,3.,1.2,1.2,'原像素',1024,1024,0)
        same(crop[4],reference[0]);same(crop[0],reference[1]);same(crop[5],reference[2])
        binary=run('SwwanMaskProcess',mask=crop[5],operation='binary',threshold=5)[0]
        fixed=run('SwwanMaskProcess',mask=binary,operation='cleanup',remove_isolated_pixels=2)[0]
        segs=run('SwwanMaskSegments',operation='from_mask',mask=fixed,drop_size=1)[0]
        selected=run('SwwanMaskSegments',operation='filter',segs=segs)[1].unsqueeze(0)
        matte=run('SwwanImageMatte',image=crop[0],mask=selected,crop_mask=False,background_color='#ffffff')
        matched=run('SwwanColorMatch',image_ref=matte[1],image_target=matte[1],match_mode='mean_std',device='cpu')[0]
        merged=run('ImageBlendSwwan',background_image=crop[0],layer_image=matched,layer_mask=selected,operation='mask_composite',invert_mask=False,opacity=75)[0]
        restored=run('SwwanRestoreCropBoxV4',background_image=self.image[:1],croped_image=merged,croped_mask=selected,crop_box=crop[2],region_info=crop[4],expand_percent=5,feather_percent=5)
        self.assertEqual(restored[0].shape,self.image[:1].shape)
        self.assertEqual(restored[1].shape,self.mask[:1].shape)


if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+rest)
