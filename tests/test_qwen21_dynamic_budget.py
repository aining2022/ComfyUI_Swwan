"""Crop-area budget: half-up rounding, context bounds, legal cap and fixed-point canvas."""
import importlib.util,sys,types,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('qwen_budget_tests');pkg.__path__=[str(ROOT/'ops')];sys.modules[pkg.__name__]=pkg
from qwen_budget_tests import qwen21_geometry as G
class Budget(unittest.TestCase):
 def test_round_half_up_and_cap(self):
  for w,h,cap,expected in [(1,1,1920,512),(528,528,1920,544),(800,800,1920,800),(1920,1920,1920,1920),(3000,3000,1920,1920),(3000,3000,1919,1888),(1600,400,1920,800)]:
   self.assertEqual(G.crop_area_resolution(w,h,cap),expected)
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
  self.assertEqual(s['native_crop_size'],(1781,1081));self.assertEqual(s['resolution'],1376)
 def test_fixed_mode_and_edge_context(self):
  im=torch.zeros(1,301,511,3);m=torch.zeros(1,301,511);m[:,:20,:10]=.6
  _,_,s=G.prepare_aligned(im,m,resolution=1024,resolution_mode='fixed');self.assertEqual(s['resolution'],1024)
  _,_,d=G.prepare_aligned(im,m,resolution_mode='crop_area',factors=(1.5,1.5,1.5,1.5));self.assertEqual(d['native_crop_size'],(15,25));self.assertEqual(d['resolution'],512)
if __name__=='__main__':unittest.main()
