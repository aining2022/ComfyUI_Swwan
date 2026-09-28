"""Outpaint units/geometry/reference and unchanged legacy branch, CPU only."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from migrate_qwen2511_workflow import load_nodes
p = argparse.ArgumentParser(); p.add_argument('--comfyui-root', type=Path, required=True)
opts, rest = p.parse_known_args()
reg = load_nodes(opts.comfyui_root)
import torch
cls = reg.NODE_CLASS_MAPPINGS['SwwanImagePadForOutpaintMasked']
path = ROOT / 'tests/fixtures/reference/outpaint_directional.py'
spec = importlib.util.spec_from_file_location('outpaint_reference', path)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


class OutpaintTests(unittest.TestCase):
    def test_reference_both_units(self):
        reference = ref.孤海外补画板按方向()
        for width, height in [(64,48),(101,79),(17,31),(192,160),(1,1)]:
            image = torch.linspace(0,1,height*width*3).reshape(1,height,width,3)
            for unit, sides in [('百分比',(50,50,0,0)),('像素',(7,11,3,9)),('百分比',(10,25,15,30))]:
                left,right,top,bottom=sides
                for alignment in [0,16,32]:
                    for feather in [0,3]:
                        with self.subTest(size=(width,height),unit=unit,sides=sides,alignment=alignment,feather=feather):
                            expected=reference.处理(image,unit,left,right,top,bottom,feather,alignment)[:2]
                            actual=cls().expand_image(image,left,top,right,bottom,feather,padding_mode='directional',padding_unit=unit,alignment=alignment)
                            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Alignment-only preserves the source crop rule (round down, full-white mask).
        image=torch.rand(1,79,101,3)
        for alignment in [8,16,32]:
            expected=reference.处理(image,'像素',0,0,0,0,0,alignment)[:2]
            actual=cls().expand_image(image,0,0,0,0,0,padding_mode='directional',alignment=alignment)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_legacy_unchanged(self):
        image=torch.rand(2,24,32,3)
        old=cls().expand_image(image,4,8,12,16,0)
        explicit=cls().expand_image(image,4,8,12,16,0,padding_mode='legacy',padding_unit='百分比',alignment=16)
        for a,b in zip(old,explicit):torch.testing.assert_close(a,b,rtol=0,atol=0)
        expected=torch.full((2,48,48,3),0.5);expected[:,8:32,4:36]=image
        expected_mask=torch.ones(2,48,48);expected_mask[:,8:32,4:36]=0
        torch.testing.assert_close(old[0],expected,rtol=0,atol=0)
        torch.testing.assert_close(old[1],expected_mask,rtol=0,atol=0)
        self.assertEqual(cls.RETURN_TYPES,('IMAGE','MASK'))

    def test_native_mask_and_errors(self):
        image=torch.rand(1,24,32,3);mask=torch.rand(1,24,32)
        actual=cls().expand_image(image,4,8,12,16,0,mask=mask,padding_mode='directional')
        torch.testing.assert_close(actual[1][:,8:32,4:36],1-mask,rtol=0,atol=0)
        self.assertTrue(torch.all(actual[1][:,:8]==1))
        crop=cls().expand_image(image,0,0,0,0,0,mask=mask,padding_mode='directional',alignment=16)
        torch.testing.assert_close(crop[1],mask[:,4:20],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'one RGB image'):
            cls().expand_image(image.repeat(2,1,1,1),0,0,0,0,0,padding_mode='directional')
        with self.assertRaisesRegex(ValueError,'empty image'):
            cls().expand_image(image[:,:2,:2],0,0,0,0,0,padding_mode='directional',alignment=16)


if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+rest)
