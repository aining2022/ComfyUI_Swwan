"""CPU acceptance for geometry, public controls, clean patches and branch wiring."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('qwen21_test_ops');pkg.__path__=[str(ROOT/'ops')];sys.modules[pkg.__name__]=pkg

def load(name):
    spec=importlib.util.spec_from_file_location(pkg.__name__+'.'+name,ROOT/'ops'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
G=load('qwen21_geometry');T=load('qwen21_tasks');R=load('qwen21_runtime');B=load('qwen21_graph')

class Geometry(unittest.TestCase):
    def test_selection_and_restore(self):
        torch.manual_seed(5)
        for shape,rect in [((301,511),(0,0,28,45)),((513,317),(300,490,317,513)),((127,211),(100,60,101,61)),((301,511),(20,70,450,85))]:
            h,w=shape;image=torch.rand(1,h,w,3);mask=torch.zeros(1,h,w)
            x0,y0,x1,y1=rect;mask[:,y0:y1,x0:x1]=.6
            # Multiple components and soft edges cannot be discarded by bounding geometry.
            mask[:,h//2:h//2+2,w//2:w//2+2]=1
            prepared,pm,seam=G.prepare_aligned(image,mask,resolution=512)
            tw,th=seam['canvas_size'];self.assertEqual(G.fixed_point(tw,th,512),(tw,th))
            bx,by,br,bb=seam['box'];ys,xs=torch.where(mask[0]>0)
            self.assertLessEqual(bx,int(xs.min()));self.assertGreater(br,int(xs.max()))
            self.assertLessEqual(by,int(ys.min()));self.assertGreater(bb,int(ys.max()))
            out,fusion=G.restore_aligned(image,torch.ones_like(prepared),seam,feather=2,expand=2)
            self.assertTrue(torch.equal(out[mask==0],image[mask==0]))
            self.assertEqual(out.shape,image.shape);self.assertGreater(float(fusion.max()),0)
    def test_coordinate_gradient_roundtrip(self):
        y,x=torch.meshgrid(torch.linspace(0,1,511),torch.linspace(0,1,317),indexing='ij')
        image=torch.stack((x,y,x*y),-1)[None];mask=torch.ones(1,511,317)
        prepared,_,seam=G.prepare_aligned(image,mask,resolution=512)
        restored,_=G.restore_aligned(image,prepared,seam)
        self.assertLess(float((restored-image).abs().max()),.004)
    def test_transparent_completion_not_clipped(self):
        image=torch.rand(1,201,317,4);image[...,3]=.8
        mask=torch.zeros(1,201,317);mask[:,70:120,110:170]=1
        prepared,_,seam=G.prepare_aligned(image,mask,resolution=512)
        gen=torch.ones_like(prepared);gen[...,3]=.5
        out,alpha=G.restore_aligned(image,gen,seam,transparent=True)
        self.assertEqual(out.shape,(1,201,317,4))
        self.assertTrue(bool(torch.any(alpha[mask==0]>0)))
        with tempfile.TemporaryDirectory() as d:
            R.save_tensor(Path(d)/'rgba.png',out)
            from PIL import Image
            with Image.open(Path(d)/'rgba.png') as im:
                self.assertEqual(im.mode,'RGBA');self.assertEqual(im.size,(317,201))
    def test_hole_fill_keeps_existing_soft_values(self):
        mask=torch.zeros(1,20,20);mask[:,3:17,3:17]=.4;mask[:,8:12,8:12]=0
        filled=G.fill_holes_preserving_softness(mask)
        self.assertEqual(float(filled[0,9,9]),1.)
        self.assertTrue(torch.equal(filled[mask>0],mask[mask>0]))
        self.assertEqual(float(filled[0,0,0]),0.)

    def test_bad_inputs(self):
        image=torch.zeros(1,101,103,3)
        for mask in (None,torch.zeros(1,101,103),torch.ones(1,99,103)):
            with self.assertRaises(ValueError):G.prepare_aligned(image,mask)

class Contracts(unittest.TestCase):
    def test_absolute_strength_and_prompt(self):
        for task in T.TASKS:
            for strength in (0,.48,1):
                c=T.resolve(task,extra_prompt='保持人物表情 / blue silk',lora_strength=strength)
                self.assertEqual(c['lora_strength'],strength)
                self.assertTrue(c['prompt'].endswith('保持人物表情 / blue silk'))
                if strength==0:self.assertIsNone(c['lora'])
        self.assertIsNone(T.resolve('edit',extra_prompt='x')['lora'])
        self.assertEqual(T.resolve('local_edit',extra_prompt='x')['lora'],T.LORAS['consistency'])
        self.assertIsNone(T.resolve('local_edit',extra_prompt='x',consistency='off')['lora'])
        self.assertIn('<image1>, high',T.face_prompt('v1.0'))
        self.assertIn('<image2>, high',T.face_prompt('v1.1'))
        with self.assertRaises(ValueError):T.resolve('edit')
    def test_schedule(self):
        for preset,n in zip(T.PRESETS,(9,8,12,16)):
            s=T.sigmas(preset);self.assertEqual(len(s),n+1);self.assertEqual(s[-1],0)
            self.assertTrue(all(a>b for a,b in zip(s,s[1:])))
        self.assertAlmostEqual(T.sigmas('experimental12')[-2],.15)
    def test_bounded_lru(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/str(i) for i in range(3)]
            for p in paths:p.write_text('weights')
            cache=R.LoraCache(24);loader=lambda p:{'w':torch.ones(4)}
            self.assertFalse(cache.get(paths[0],loader)[1]);self.assertTrue(cache.get(paths[0],loader)[1])
            cache.get(paths[1],loader);self.assertLessEqual(cache.bytes,24)
            self.assertFalse(cache.get(paths[0],loader)[1])
    def test_person_failure_keeps_diagnostics(self):
        from unittest.mock import patch
        modules={name:types.ModuleType(name) for name in ('folder_paths','comfy','comfy.sd','comfy.utils','comfy.samplers','comfy.sample','latent_preview','nodes')}
        modules['comfy'].__path__=[]
        for name in ('sd','utils','samplers','sample'):setattr(modules['comfy'],name,modules['comfy.'+name])
        modules['comfy.samplers'].ksampler=lambda name:object()
        modules['comfy.sample'].prepare_noise=lambda *args:torch.zeros(1)
        def fail(*args,**kwargs):raise RuntimeError('injected person 2 sampling failure')
        modules['comfy.sample'].sample_custom=fail
        modules['latent_preview'].prepare_callback=lambda *args:None
        modules['nodes'].NODE_CLASS_MAPPINGS={'QwenImage21SageAttentionT8':types.SimpleNamespace(execute=lambda m,**kwargs:(m,))}
        model=types.SimpleNamespace(clone=lambda:object())
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,modules),patch.object(R,'encode',return_value=([],{'samples':torch.zeros(1,16,1,32,32)})):
            config=dict(T.resolve('face_swap',lora_strength=0),artifact_dir=d,run_id='failure-check')
            before=torch.zeros(1,32,32,3)
            with self.assertRaisesRegex(RuntimeError,'person-02 failed'):
                R.infer(model,None,None,before,[],config,person=1,previous=before)
            record=json.loads(Path(d,'failure.json').read_text())
            self.assertEqual(record['person'],2);self.assertEqual(record['status'],'failed')
            self.assertTrue(Path(d,'person-02-before.png').exists())
            self.assertFalse(Path(d,'result.json').exists())
            self.assertFalse(Path(d,'person-03.json').exists())

    def test_row_major_mapping(self):
        from types import SimpleNamespace
        # Deliberately column-major input, including slight vertical staggering.
        items=[SimpleNamespace(bbox=(x,y,x+70,y+100),name=(x,y)) for x,y in [(0,0),(0,500),(300,7),(300,510),(600,2),(600,502)]]
        ordered=B.row_order(((1024,960),items))[1]
        self.assertEqual([s.name for s in ordered],[(0,0),(300,7),(600,2),(0,500),(300,510),(600,502)])
        self.assertEqual([s.name for s in items],[(0,0),(0,500),(300,7),(300,510),(600,2),(600,502)])

    def test_face_wiring(self):
        c=T.resolve('face_swap',people=6,extra_prompt='smile',lora_strength=.48)
        graph=B.face_graph('MODEL','CLIP','VAE','IMAGE','MASK',['REF']*6,c)
        kinds={n['class_type'] for n in graph.values()}
        for bad in ('UNETLoader','KSampler','QwenImage21BlockCacheT8','QwenImage21SpectrumT8','EditTextEncode_EditUtils'):
            self.assertNotIn(bad,kinds)
        self.assertEqual(graph['131']['inputs']['crop_mode'],'qwen21_aligned')
        self.assertEqual(graph['247']['inputs']['references'],['reference',0])
        self.assertEqual(graph['63']['inputs']['background'],['72',2])
        self.assertEqual(graph['158']['inputs']['image'],['140',0])
        self.assertEqual(graph['202']['inputs']['index'],['72',1])
        self.assertNotIn('138',graph)
        self.assertEqual(graph['110']['inputs']['segs'],['233',0])
        self.assertEqual(graph['233']['class_type'],'SwwanQwen21FaceRegions')

if __name__=='__main__':unittest.main()
