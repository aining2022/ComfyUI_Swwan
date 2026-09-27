"""Regression of the user's edited UI graph; no Downloads or sibling plugins required."""
import copy
import json
import unittest
import test_face_head_processing as ref
from repair_qwen_face_head_workflow import repair
from workflow_contracts import validate_workflow

ROOT, registry = ref.ROOT, ref.registry


class RepairRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT/'tests/fixtures/qwen-face-head-current.json').read_text())
        cls.fixed = repair(cls.original, registry)
        cls.nodes = {n['id']:n for n in cls.fixed['nodes']}

    def test_pruning_and_preservation(self):
        before = copy.deepcopy(self.original)
        self.assertEqual((len(self.fixed['nodes']),len(self.fixed['links'])),(145,167))
        self.assertEqual(self.fixed,repair(self.fixed,registry))
        self.assertEqual(self.fixed,json.loads((ROOT/'examples/qwen-face-head-final-swwan.json').read_text()))
        self.assertEqual(self.original,before)
        deleted = {'PreviewImage','SwwanImageAndMaskPreview','GuhaiBatchProgress_孤海批处理进度条','CachePreviewBridge','MarkdownNote','忽略多组孤海'}
        self.assertFalse(any(n['type'] in deleted for n in self.fixed['nodes']))
        for node in self.fixed['nodes']:
            old = next(n for n in self.original['nodes'] if n['id']==node['id'])
            for key in ('pos','size','mode','title','properties'):
                self.assertEqual(node.get(key),old.get(key),(node['id'],key))
            if node['type'] not in registry.NODE_CLASS_MAPPINGS:
                self.assertEqual(node.get('widgets_values'),old.get('widgets_values'))
            if node['type'] in ('easy ifElse','easy anythingIndexSwitch'):
                self.assertEqual(node['inputs'],old['inputs']) # every selectable branch survives
        self.assertEqual(self.nodes[163]['widgets_values'],['换脸_无高清'])
        loop = {i['name']:i.get('link') for i in self.nodes[162]['inputs']}
        self.assertIsNone(loop['initial_value2']); self.assertIsNone(loop['initial_value3'])
        self.assertIsNotNone(loop['initial_value1']); self.assertIsNotNone(loop['flow'])
        link = next(l for l in self.fixed['links'] if l[0]==160)
        self.assertEqual(link[1:5],[71,0,136,0])
        validate_workflow(self.fixed,registry)

    def test_repaired_parameters(self):
        draw = self.nodes[62]['widgets_values_named']
        self.assertEqual(draw,{'color':'255, 255, 255','opacity':1.0,'device':'cpu'})
        p = self.nodes[92]['widgets_values_named']
        for name,value in dict(width=768,height=768,resize_mode='essentials',essentials_method='keep proportion',essentials_condition='always',essentials_interpolation='lanczos',divisible_by=0).items(): self.assertEqual(p[name],value)
        for nid in (47,58): self.assertEqual(self.nodes[nid]['widgets_values_named']['background_color'],'#ffffff')
        edited = copy.deepcopy(self.fixed)
        next(n for n in edited['nodes'] if n['id']==62)['widgets_values_named']['opacity']=.35
        next(n for n in edited['nodes'] if n['id']==47)['widgets_values_named']['background_color']='#123456'
        again = {n['id']:n for n in repair(edited,registry)['nodes']}
        self.assertEqual(again[62]['widgets_values_named']['opacity'],.35)
        self.assertEqual(again[47]['widgets_values_named']['background_color'],'#123456')

    def test_strict_validation(self):
        for name,value in [('opacity','cpu'),('opacity',True),('device','cuda'),('opacity',float('nan'))]:
            bad = copy.deepcopy(self.fixed)
            next(n for n in bad['nodes'] if n['id']==62)['widgets_values_named'][name]=value
            with self.assertRaises(ValueError): repair(bad,registry)
        bad = copy.deepcopy(self.fixed)
        next(n for n in bad['nodes'] if n['id']==62)['inputs'][0]['link']=None
        with self.assertRaises(ValueError): validate_workflow(bad,registry)
        bad = copy.deepcopy(self.fixed)
        next(n for n in bad['nodes'] if n['type']=='GetNode')['widgets_values']=['no setter']
        with self.assertRaisesRegex(ValueError,'Unresolved'): repair(bad,registry)

    def test_draw_old_interfaces(self):
        for values,opacity,device in [(['255, 255, 255','cpu'],1.,'cpu'),(['255, 255, 255',.4,'gpu'],.4,'gpu')]:
            original = copy.deepcopy(ref.FaceHeadProcessing.original) if hasattr(ref.FaceHeadProcessing,'original') else json.loads((ROOT/'tests/fixtures/qwen-face-head-original.json').read_text())
            node = next(n for n in original['nodes'] if n['id']==62); node['widgets_values']=values
            fixed = next(n for n in ref.migrate(original,registry)['nodes'] if n['id']==62)
            self.assertEqual(fixed['widgets_values_named'],dict(color='255, 255, 255',opacity=opacity,device=device))
        node['widgets_values']=['255, 255, 255','cpu','cpu']
        with self.assertRaisesRegex(ValueError,'opacity'): ref.migrate(original,registry)

    def test_current_pixels_and_restore_position(self):
        torch = ref.torch
        image = torch.zeros(1,48,64,3); mask = torch.zeros(1,48,64); mask[:,12:28,20:40]=1
        def run(nid, **kwargs):
            node = self.nodes[nid]; params = dict(node['widgets_values_named']); params.update(kwargs)
            return ref.run(node['type'],**params)
        drawn = run(62,image=image,mask=mask)[0]
        torch.testing.assert_close(drawn,mask.unsqueeze(-1).expand_as(image),rtol=0,atol=0)
        resized = run(92,image=drawn)[0]
        expected = ref.refs['ImageResize']().execute(drawn,768,768,'keep proportion','lanczos','always',0)[0]
        ref.same(resized,expected)
        matte = run(47,image=image,mask=mask)
        self.assertEqual(matte[0].shape[-1],4)
        self.assertTrue(torch.all(matte[1][0,0,0]==1))
        crop = run(91,image=image,mask=mask)
        simulated = torch.ones_like(crop[0])*.5
        restored = run(63,background_image=image,croped_image=simulated,croped_mask=crop[5],crop_box=crop[2],region_info=crop[4])
        self.assertEqual(restored[0].shape,image.shape)
        # The reference seam restoration is authoritative for expansion/feathering.
        reference = __import__('importlib').import_module(registry.__name__+'.edit_region').EditRegionRestore()
        expected = reference.恢复图像(crop[4],simulated,self.nodes[63]['widgets_values_named']['expand_percent'],self.nodes[63]['widgets_values_named']['feather_percent'],crop[5],image)
        ref.same(restored[0],expected[0])


if __name__ == '__main__': unittest.main(argv=[__file__]+ref.rest)
