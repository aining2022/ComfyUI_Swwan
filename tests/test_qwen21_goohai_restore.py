"""CPU parity with native GH restoration: same coordinates, colors and protection."""
import importlib.util,sys,types,unittest,tempfile
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('qwen_gh_test');pkg.__path__=[str(ROOT/'ops')];sys.modules[pkg.__name__]=pkg
from qwen_gh_test import qwen21_geometry as G,qwen21_gh_seam as S
spec=importlib.util.spec_from_file_location('gh_reference',ROOT/'tests/fixtures/reference/gh_crop_restore_color_8d9499.py');GH=importlib.util.module_from_spec(spec);spec.loader.exec_module(GH)
class Geometry(unittest.TestCase):
 def test_odd_edge_small_multi_and_soft_masks(self):
  for h,w,rect in [(301,511,(0,0,28,45)),(513,317,(300,490,317,513)),(127,211,(100,60,101,61)),(301,511,(20,70,450,85))]:
   y,x=torch.meshgrid(torch.linspace(.1,.8,h),torch.linspace(.1,.8,w),indexing='ij');im=torch.stack((x,y,(x+y)/2),-1)[None];m=torch.zeros(1,h,w)
   x0,y0,x1,y1=rect;m[:,y0:y1,x0:x1]=.6;m[:,h//2:h//2+2,w//2:w//2+2]=1
   prepared,pm,seam=G.prepare_aligned(im,m,512);gh=S.to_goohai_seam(seam);tw,th=seam['canvas_size'];self.assertEqual((tw,th),G.fixed_point(tw,th,512));self.assertEqual(gh['目标尺寸'],(th,tw))
   ys,xs=torch.where(m[0]>0);vy0,vy1,vx0,vx1=gh['裁剪区域'];self.assertLessEqual(vx0,int(xs.min()));self.assertGreater(vx1,int(xs.max()));self.assertLessEqual(vy0,int(ys.min()));self.assertGreater(vy1,int(ys.max()))
   restored=GH.Restore().恢复图像(gh,prepared,0,0,pm,颜色还原强度=0)[0]
   final=torch.where((m>0)[...,None],restored,im)
   self.assertLess(float((final-im).abs().max()),.02)
   self.assertTrue(torch.equal(final[m==0],im[m==0]))
   protect=torch.zeros_like(m);protect[:,h//2:h//2+1,w//2:w//2+2]=1
   gated=torch.where(((m>0)&(protect==0))[...,None],restored,im);self.assertTrue(torch.equal(gated[protect==1],im[protect==1]))
   with tempfile.TemporaryDirectory() as d:
    a=(gated[0].numpy()*255).astype(np.uint8);Image.fromarray(a).save(Path(d)/'x.png');self.assertTrue(np.array_equal(a,np.array(Image.open(Path(d)/'x.png'))))
 def test_native_color_controls(self):
  y,x=torch.meshgrid(torch.linspace(.1,.7,101),torch.linspace(.1,.7,173),indexing='ij');im=torch.stack((x,y,(x+y)/2),-1)[None];m=torch.ones(1,101,173)
  p,pm,s=G.prepare_aligned(im,m,512);gh=S.to_goohai_seam(s);raw=p+torch.tensor([.025,-.025,.018]);r=GH.Restore()
  off=r.恢复图像(gh,raw,0,0,pm,颜色还原强度=0)[0];on=r.恢复图像(gh,raw,0,0,pm,颜色还原强度=1)[0]
  self.assertLess(float((on-im).abs().mean()),float((off-im).abs().mean())*.3)
 def test_legacy_record_and_empty_guard(self):
  with self.assertRaises(ValueError):S.to_goohai_seam({})
  with self.assertRaises(ValueError):G.prepare_aligned(torch.zeros(1,17,33,3),torch.zeros(1,17,33),512)
if __name__=='__main__':unittest.main()
