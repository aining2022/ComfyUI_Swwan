"""Crop-area budget: half-up rounding, context bounds, legal cap and fixed-point canvas."""
import importlib.util,sys,types,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('qwen_budget_tests');pkg.__path__=[str(ROOT/'ops')];sys.modules[pkg.__name__]=pkg
from qwen_budget_tests import qwen21_geometry as G
class Budget(unittest.TestCase):
 def test_configurable_minimum_and_alignment(self):
  for minimum,cap,expected in [(32,1920,32),(256,1920,256),(257,1920,288),(384,1920,384),(513,1920,544),(768,1920,768),(1560,1600,1568),(2048,2048,2048)]:
   with self.subTest(minimum=minimum,cap=cap):
    self.assertEqual(G.crop_area_resolution(1,1,cap,minimum),expected)
  self.assertEqual(G.crop_area_resolution(800,800,1920,256),800)
  self.assertEqual(G.crop_area_resolution(3000,3000,1560,256),1536)
  for minimum,cap in [(0,1920),(31,1920),(2049,2048),(1024,768),(513,543)]:
   with self.subTest(minimum=minimum,cap=cap),self.assertRaises(ValueError):G.crop_area_resolution(100,100,cap,minimum)
 def test_small_budgets_are_fixed_points_and_restore(self):
  y,x=torch.meshgrid(torch.linspace(0,1,101),torch.linspace(0,1,173),indexing='ij')
  im=torch.stack((x,y,x*y),-1)[None];m=torch.zeros(1,101,173);m[:,:12,:20]=.6
  for minimum in (32,128,256,384,768):
   with self.subTest(minimum=minimum):
    p,pm,s=G.prepare_aligned(im,m,factors=(1,1,1,1),resolution_mode='crop_area',min_resolution=minimum)
    self.assertEqual(s['resolution'],minimum);self.assertEqual(s['min_resolution'],minimum)
    self.assertEqual(G.fixed_point(*s['canvas_size'],minimum),s['canvas_size'])
    self.assertEqual(p.shape[1:3],pm.shape[1:3]);self.assertEqual(s['encoder_resolution'],0)
    out,_=G.restore_aligned(im,p,s);self.assertEqual(out.shape,im.shape)
    self.assertTrue(torch.equal(out[m==0],im[m==0]));self.assertLess(float((out-im).abs().max()),.01)
 def test_default_minimum_and_fixed_mode_compatibility(self):
  im=torch.rand(1,101,173,3);m=torch.zeros(1,101,173);m[:,10:40,20:60]=1
  old=G.prepare_aligned(im,m,resolution_mode='crop_area')
  explicit=G.prepare_aligned(im,m,resolution_mode='crop_area',min_resolution=128,max_resolution=1024)
  for a,b in zip(old[:2],explicit[:2]):self.assertTrue(torch.equal(a,b))
  self.assertEqual(old[2]['canvas_size'],explicit[2]['canvas_size']);self.assertEqual(old[2]['resolution'],128)
  self.assertEqual(G.crop_area_resolution(1,1),128);self.assertEqual(G.crop_area_resolution(3000,3000),1024)
  fixed=G.prepare_aligned(im,m,resolution=512,min_resolution=1024,max_resolution=768)
  self.assertEqual(fixed[2]['resolution'],512)
 def test_round_half_up_and_cap(self):
  for w,h,cap,expected in [(1,1,1920,512),(528,528,1920,544),(800,800,1920,800),(1920,1920,1920,1920),(3000,3000,1920,1920),(3000,3000,1919,1888),(1600,400,1920,800)]:
   self.assertEqual(G.crop_area_resolution(w,h,cap,512),expected)
  with self.assertRaises(ValueError):G.crop_area_resolution(100,100,500)
 def test_budget_uses_expanded_bbox_before_ratio_adjustment(self):
  im=torch.zeros(1,1500,2000,3);m=torch.zeros(1,1500,2000);m[:,400:800,400:1000]=1
  p,pm,s=G.prepare_aligned(im,m,factors=(1.5,1.25,1.5,1.25),resolution_mode='crop_area')
  self.assertEqual(s['native_crop_size'],(900,700));self.assertEqual(s['context_box'],(200,200,1100,900));self.assertEqual(s['resolution'],800)
  self.assertNotEqual(s['resolution'],G.crop_area_resolution(2000,1500));self.assertNotEqual(s['resolution'],G.crop_area_resolution(600,400))
  self.assertEqual(G.fixed_point(*s['canvas_size'],800),s['canvas_size']);self.assertEqual(p.shape[1:3],pm.shape[1:3])
 def test_disconnected_support_is_not_white_pixel_area(self):
  im=torch.zeros(1,1200,2000,3);m=torch.zeros(1,1200,2000);m[:,20,20]=1;m[:,1100,1800]=1
  _,_,s=G.prepare_aligned(im,m,factors=(1,1,1,1),resolution_mode='crop_area')
  self.assertEqual(s['native_crop_size'],(1781,1081));self.assertEqual(s['resolution'],1024)
 def test_fixed_mode_and_edge_context(self):
  im=torch.zeros(1,301,511,3);m=torch.zeros(1,301,511);m[:,:20,:10]=.6
  _,_,s=G.prepare_aligned(im,m,resolution=1024,resolution_mode='fixed');self.assertEqual(s['resolution'],1024)
  _,_,d=G.prepare_aligned(im,m,resolution_mode='crop_area',factors=(1.5,1.5,1.5,1.5));self.assertEqual(d['native_crop_size'],(15,25));self.assertEqual(d['resolution'],128)
if __name__=='__main__':unittest.main()
