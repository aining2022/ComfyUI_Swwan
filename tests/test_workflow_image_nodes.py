"""CPU contract and reference checks; run with the ComfyUI Python interpreter.

python tests/test_workflow_image_nodes.py --comfyui-root PATH
References are frozen inside tests/fixtures. No sibling plugins or git HEAD are required.
"""
import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from migrate_qwen2511_workflow import load_nodes, defaults, MAPPINGS

parser = argparse.ArgumentParser()
parser.add_argument("--comfyui-root", type=Path, required=True)
opts, unittest_args = parser.parse_known_args()
registry = load_nodes(opts.comfyui_root)
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from collections import Counter
import colorsys
import re

C = registry.NODE_CLASS_MAPPINGS
reference = dict(torch=torch, np=np, cv2=cv2, Image=Image, ImageFilter=ImageFilter,
                 math=math, Counter=Counter, colorsys=colorsys, re=re,
                 HIGH_QUALITY_INTERPOLATION=Image.LANCZOS, MASK_INTERPOLATION=Image.BILINEAR,
                 main_device=torch.device("cpu"), tqdm=lambda x, **kw: x)
refs_available = True
sources = sorted((ROOT / "tests/fixtures/reference").glob("*.py"))
for path in sources:
    classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), reference)


def invoke(node_id, **kwargs):
    cls = C[node_id]
    values = defaults(cls)
    values.update(kwargs)
    if "hidden" in cls.INPUT_TYPES(): values.setdefault("unique_id", None)
    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(cls(), cls.FUNCTION)(**values)


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, Image.Image): np.testing.assert_array_equal(np.array(a), np.array(b))
    else: assert a == b, (a, b)


class WorkflowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(4)
        cls.image = torch.randint(0, 256, (1, 96, 128, 3)).float() / 255
        cls.mask = torch.zeros(1, 96, 128)
        cls.mask[:, 28:68, 36:92] = 1

    def test_registry_and_workflow(self):
        migrated = json.loads((ROOT / "examples/qwen2511-remove-single-swwan.json").read_text())
        nodes = {n["id"]: n for n in migrated["nodes"]}
        original=json.loads((ROOT/'tests/fixtures/qwen2511-original.json').read_text())
        import copy
        for node in original['nodes']:
            got=copy.deepcopy(nodes[node['id']])
            self.assertEqual(node['pos'],got['pos']);self.assertEqual(node['size'],got['size'])
            if node['type'] not in MAPPINGS:
                # LoadImage receives the explicitly required extra original-image link.
                if node['type']=='LoadImage':got['outputs']=node['outputs']
                self.assertEqual(got,node)
        self.assertEqual(len(set(n["type"] for n in nodes.values()) & set(MAPPINGS)), 0)
        for link_id, source, slot, target, index, kind in migrated["links"]:
            self.assertEqual(nodes[source]["outputs"][slot]["type"], kind)
            self.assertEqual(nodes[target]["inputs"][index]["type"], kind)
            self.assertIn(link_id, nodes[source]["outputs"][slot]["links"])
            self.assertEqual(nodes[target]["inputs"][index]["link"], link_id)
        for node in nodes.values():
            if node["type"] not in C: continue
            cls = C[node["type"]]
            self.assertEqual(len(node["widgets_values"]), len(defaults(cls)))
            values = dict(zip(defaults(cls), node["widgets_values"]))
            self.assertEqual(node["widgets_values_named"], values)
            schema = {**cls.INPUT_TYPES().get("required", {}), **cls.INPUT_TYPES().get("optional", {})}
            for name, val in values.items():
                if isinstance(schema[name][0], list): self.assertIn(val, schema[name][0])
            for name, definition in cls.INPUT_TYPES().get("required", {}).items():
                if name not in values:
                    self.assertTrue(any(i["name"] == name and i.get("link") is not None for i in node["inputs"]))
        self.assertEqual(C["SwwanCropByMaskV5"].RETURN_TYPES[:4], ("IMAGE", "IMAGE", "BOX", "IMAGE"))
        for node_id in ["LayerUtility: CropByMask V2", "LayerUtility: CropByMask V3", "LayerUtility: CropByMask V4",
                        "LayerUtility: RestoreCropBox", "LayerUtility: RestoreCropBox V2", "SwwanRestoreCropBoxV3", "ImageResizeKJ"]:
            self.assertEqual(C[json.loads((ROOT/"tests/fixtures/legacy_id_map.json").read_text()).get(node_id,node_id)].CATEGORY, "Swwan/Legacy")
        self.assertIn("SwwanDrawMaskOnImage", C)  # pre-existing user edit

    def test_legacy_regression_against_base(self):
        # Read the pre-change classes for this worktree's targeted regression check.
        selected = {
            "crop_by_mask_v5.py": ["CropByMaskV5"],
            "restore_crop_box_v4.py": ["RestoreCropBoxV4"],
            "image_blend.py": ["ImageBlendSwwan"],
            "image_nodes.py": ["ImageResizeKJv2", "ImageResizeByMegapixels"],
        }
        baselines = {}
        for filename, names in selected.items():
            mod = importlib.import_module(registry.__name__ + "." + filename[:-3])
            ns = dict(vars(mod))
            source = (ROOT / "tests/fixtures" / ("baseline_" + filename)).read_text()
            classes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
            exec(compile(ast.Module(body=classes, type_ignores=[]), filename, "exec"), ns)
            for name in names: baselines[name] = ns[name]
        cases = [
            ("SwwanCropByMaskV5", "CropByMaskV5", dict(image=self.image.repeat(2,1,1,1), mask_image=self.mask.unsqueeze(-1).repeat(1,1,1,3)), 4),
            ("SwwanRestoreCropBoxV4", "RestoreCropBoxV4", dict(background_image=self.image.repeat(2,1,1,1), croped_image=self.image[:,20:60,30:80], crop_box=[30,20,80,60], feathering=3, device="CPU"), 2),
            ("ImageBlendSwwan", "ImageBlendSwwan", dict(background_image=self.image, layer_image=1-self.image, layer_mask=self.mask, invert_mask=False), 1),
            ("ImageResizeKJv2Alternative", "ImageResizeKJv2", dict(image=self.image, width=80, height=64, keep_proportion="crop", upscale_method="bilinear", mask=self.mask), 4),
            ("ImageResizeByMegapixels", "ImageResizeByMegapixels", dict(image=self.image, megapixels=0.02, aspect_ratio="4:3", upscale_method="bilinear", mask=self.mask), 5),
        ]
        for node_id, name, kwargs, slots in cases:
            with self.subTest(node=node_id), contextlib.redirect_stdout(io.StringIO()):
                cls = baselines[name]; values = defaults(cls); values.update(kwargs)
                if "hidden" in cls.INPUT_TYPES(): values["unique_id"] = None
                expected = getattr(cls(), cls.FUNCTION)(**values)
                same(invoke(node_id, **kwargs)[:slots], expected)
                # Older API prompts omit every newly added optional parameter.
                current = C[node_id]
                same(getattr(current(), current.FUNCTION)(**values)[:slots], expected)
        with self.subTest(megapixels_bypass=True):
            output = invoke("ImageResizeByMegapixels", image=self.image, megapixels=0)
            same(output[0], self.image)

    def test_crop_restore_reference(self):
        masks = {"normal": self.mask, "empty": torch.zeros_like(self.mask), "hole": self.mask.clone(),
                 "edge": torch.zeros_like(self.mask), "tiny": torch.zeros_like(self.mask)}
        masks["hole"][:,40:52,50:70] = 0
        masks["edge"][:,0:28,0:40] = 1
        masks["tiny"][:,40:42,50:52] = 1
        for name, mask in masks.items():
            for size in ["原像素", "原像素1：1", "自定义宽高"]:
                for alignment in [0, 16]:
                    with self.subTest(mask=name, size=size, alignment=alignment):
                        params = dict(image=self.image, mask=mask, crop_mode="edit_region", fill_mask_holes=True,
                                      top_reserve_ratio=1.3-1, bottom_reserve_ratio=1.3-1,
                                      left_reserve_ratio=1.3-1, right_reserve_ratio=1.3-1,
                                      output_size=size, custom_width=96, custom_height=64, alignment=alignment)
                        got = invoke("SwwanCropByMaskV5", **params)
                        expected = reference["孤海_遮罩裁剪V2"]().裁剪图像(self.image, mask, True, 1.3,1.3,1.3,1.3,size,96,64,alignment)
                        same(got[4], expected[0]); same(got[0], expected[1]); same(got[5], expected[2])
                        processed = (1-got[0]).flip(2)
                        for supplied in [None, got[5], torch.zeros_like(got[5])]:
                            result = invoke("SwwanRestoreCropBoxV4", background_image=self.image, croped_image=processed,
                                            crop_box=got[2], region_info=got[4], croped_mask=supplied, expand_percent=4, feather_percent=3)
                            restored = reference["孤海_裁剪恢复"]().恢复图像(expected[0], processed,4,3,supplied,self.image)
                            same(result[0], restored[0]); self.assertEqual(result[1].shape, self.mask.shape)
                            self.assertTrue(torch.isfinite(result[0]).all())
        # A zero custom dimension means the supplied dimension is the longest edge.
        got = invoke("SwwanCropByMaskV5", image=self.image, mask=self.mask, crop_mode="edit_region",
                     output_size="自定义宽高", custom_width=0, custom_height=64)
        self.assertEqual(max(got[0].shape[1:3]), 64)

    def test_restore_mask_is_actual_composite_region(self):
        for edge in [False, True]:
            with self.subTest(edge=edge):
                mask = self.mask.clone()
                if edge:
                    mask.zero_(); mask[:, :24, :32] = 1
                crop = invoke("SwwanCropByMaskV5", image=torch.zeros_like(self.image), mask=mask,
                              crop_mode="edit_region", top_reserve_ratio=0.3, left_reserve_ratio=0.3)
                restored, full_mask = invoke("SwwanRestoreCropBoxV4", background_image=torch.zeros_like(self.image),
                                             croped_image=torch.ones_like(crop[0]), crop_box=crop[2],
                                             region_info=crop[4], expand_percent=4, feather_percent=3)
                self.assertTrue(torch.all((full_mask >= 0) & (full_mask <= 1)))
                same(restored[..., 0], full_mask)
                same(restored[..., 1], full_mask)
                same(restored[..., 2], full_mask)

    def test_resize_and_range_reference(self):
        method_map = {"bilinear":"双线性插值", "bicubic":"双三次插值", "area":"区域", "nearest-exact":"邻近-精确", "lanczos":"Lanczos"}
        for fit in ["拉伸", "裁剪", "填充_自定颜色", "填充_边框颜色", "填充_边缘像素", "总像素_等比例"]:
            for method, ref_method in method_map.items():
                for alignment in [0, 16]:
                    with self.subTest(fit=fit, method=method, alignment=alignment):
                        got = invoke("ImageResizeKJv2Alternative", image=self.image, mask=self.mask,
                                     width=80,height=80, resize_mode="edit_size",size_rule="自定义宽高",
                                     edit_fit=fit,upscale_method=method,divisible_by=alignment,fill_color="#364254")
                        expected = reference["图像缩放V2_孤海"]().执行缩放(self.image,"自定义宽高",80,80,1024,ref_method,fit,"居中","总是","#364254",alignment,self.mask)
                        same(got, expected)
        for rule in ["按长边等比例", "按短边等比例"]:
            for condition in ["总是", "最长边大于时", "最小边小于时"]:
                with self.subTest(rule=rule,condition=condition):
                    got=invoke("ImageResizeKJv2Alternative",image=self.image,resize_mode="edit_size",size_rule=rule,
                               edge_length=64,execute_condition=condition,upscale_method="lanczos",divisible_by=16)
                    expected=reference["图像缩放V2_孤海"]().执行缩放(self.image,rule,512,512,64,"Lanczos","裁剪","居中",condition,"#364254",16)
                    same(got,expected)
        for mode in ["长边", "短边", "宽度", "高度", "宽度与高度"]:
            for im, ma in [(self.image,self.mask),(self.image,None),(None,self.mask)]:
                with self.subTest(range_mode=mode,image=im is not None):
                    got=invoke("SwwanImageResizeRange",图像=im,遮罩=ma,限制模式=mode,最小尺寸=64,最大尺寸=112,整除数=16)
                    expected=reference["图像缩放范围孤海"]().resize_image_range(im,ma,mode,64,112,16)
                    same(got,expected)
        black=invoke("SwwanImageResizeRange",遮罩=torch.zeros_like(self.mask),最小尺寸=64,最大尺寸=112,整除数=16)
        self.assertIsNone(black[0]);self.assertIsInstance(black[1],torch.Tensor);self.assertEqual(float(black[1].sum()),0)

    def test_mask_color_composite_and_rgb(self):
        for mask in [self.mask,torch.zeros_like(self.mask),self.mask*.4]:
            for block in [8,16,128]:
                with self.subTest(block=block):
                    same(invoke("SwwanBlockifyMask",masks=mask,block_size=block),reference["BlockifyMask"]().process(mask,block,"cpu"))
        color=invoke("LayerUtility: ColorImage (Swwan)",width=40,height=32,color="#00ff00")[0]
        for expand,blur,match in [(0,0,True),(2,3,True),(-2,1,False)]:
            for opacity in [0,75,100]:
                with self.subTest(expand=expand,opacity=opacity):
                    got=invoke("ImageBlendSwwan",background_image=self.image,layer_image=color,layer_mask=self.mask,
                               invert_mask=False,operation="mask_composite",mask_expand=expand,mask_blur=blur,
                               match_image_size=match,opacity=opacity)
                    expected=reference["GulfSeaImageMergeMask"]().merge_images(self.image,color,opacity/100,expand,blur,match,self.mask)
                    same(got,expected)
        for color in ["364254","#abc","255,0,128","（120，50%，25%）","0.1,0.2,0.3","invalid"]:
            for fmt in ["#HEX","HEX","RGB","HSL"]:
                same(invoke("SwwanColorConverter",色值=color,转换后=fmt),reference["ColorConverterGuhai"]().convert_color(color,fmt))
        path = ROOT / "tests/fixtures/was_tensors.py"
        spec = importlib.util.spec_from_file_location("was_tensor_reference", path)
        tensors = importlib.util.module_from_spec(spec); spec.loader.exec_module(tensors)
        for im in [self.image.repeat(2,1,1,1),self.image[...,0:1],self.image[...,0],
                   torch.cat([self.image,self.mask.unsqueeze(-1)],dim=-1),self.image*2,self.image-.1]:
            same(invoke("SwwanImagesToRGB",images=im)[0],tensors.filtered_planes(im,lambda p:p.convert("RGB")))

    def test_migrated_processing_chain(self):
        workflow=json.loads((ROOT/"examples/qwen2511-remove-single-swwan.json").read_text())
        by_id={n["id"]:n for n in workflow["nodes"]}
        output={89:(self.image,self.mask)}
        for id in [305,380,309,308,503,323,486,506,505]:
            node=by_id[id]
            if id==486:
                im=output[323][0];output[id]=(im.shape[2],im.shape[1],im.shape[0]);continue
            params=dict(zip(defaults(C[node["type"]]),node["widgets_values"]))
            for inp in node["inputs"]:
                if inp.get("link") is not None:
                    link=next(l for l in workflow["links"] if l[0]==inp["link"])
                    params[inp["name"]]=output[link[1]][link[2]]
            output[id]=invoke(node["type"],**params)
        output[216]=(1-output[505][0],)
        node=by_id[314];params=dict(zip(defaults(C[node["type"]]),node["widgets_values"]))
        for inp in node["inputs"]:
            if inp.get("link") is not None:
                link=next(l for l in workflow["links"] if l[0]==inp["link"])
                params[inp["name"]]=output[link[1]][link[2]]
        restored=invoke(node["type"],**params)
        self.assertEqual(restored[0].shape,self.image.shape)
        self.assertTrue(torch.isfinite(restored[0]).all())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]]+unittest_args)
