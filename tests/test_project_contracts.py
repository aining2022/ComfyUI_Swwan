"""CPU remediation acceptance, including actual files and migration behavior."""
import argparse
import ast
import copy
import importlib
import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from migrate_qwen2511_workflow import load_nodes,defaults
from migrate_workflow import migrate,ALIASES
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True)
opts,rest=p.parse_known_args();reg=load_nodes(opts.comfyui_root)
import torch
import numpy as np
from PIL import Image
import folder_paths
C=reg.NODE_CLASS_MAPPINGS

def run(key,**values):
    cls=C[key];args=defaults(cls);args.update(values)
    for name in cls.INPUT_TYPES().get('hidden',{}):args.setdefault(name,None)
    return getattr(cls(),cls.FUNCTION)(**args)

def plain(value):return json.loads(json.dumps(value,default=str))

class Acceptance(unittest.TestCase):
    def setUp(self):
        self.image=torch.arange(2*32*40*3).reshape(2,32,40,3).float().remainder(256)/255
        self.mask=torch.zeros(1,32,40);self.mask[:,4:20,5:30]=1
    def test_registry_and_old_contracts(self):
        manifest=reg.registry.MANIFEST
        self.assertEqual(sum(x['tier']=='primary' for x in manifest),21)
        self.assertEqual(len(C),len(json.loads((ROOT/"node_manifest.json").read_text())));self.assertFalse(set(ALIASES)&C.keys())
        self.assertIn('ImageResizeByMegapixels',C);self.assertIn('SwwanDrawMaskOnImage',C)
        with self.assertRaisesRegex(ValueError,'Duplicate Swwan'):
            reg.registry.build_registry(reg.__name__,[manifest[0],manifest[0]])
        for cls in C.values():
            for group in ['required','optional']:
                for data in cls.INPUT_TYPES().get(group,{}).values():
                    if isinstance(data[0],list) and len(data)>1 and 'default' in data[1]:self.assertIn(data[1]['default'],data[0])
        old=json.loads((ROOT/'tests/fixtures/pre_remediation_contracts.json').read_text())
        for key,contract in old.items():
            new=ALIASES.get(key,key);cls=C[new];schema=plain(cls.INPUT_TYPES())
            with self.subTest(node=key):
                self.assertEqual(list(cls.RETURN_TYPES)[:len(contract['outputs'])],contract['outputs'])
                self.assertEqual(getattr(cls,'INPUT_IS_LIST',False),contract['input_is_list'])
                self.assertEqual(list(getattr(cls,'OUTPUT_IS_LIST',[])),contract['output_is_list'])
                for group,inputs in contract['inputs'].items():
                    for name,data in inputs.items():
                        got=schema[group][name]
                        if isinstance(data,list) and isinstance(data[0],list):
                            self.assertEqual(got[0][:len(data[0])],data[0])
                            if new=='ImageResizeKJv2Alternative' and name=='keep_proportion':
                                self.assertEqual(data[1]['default'],False);self.assertEqual(got[1]['default'],'stretch')
                                self.assertEqual({k:v for k,v in got[1].items() if k!='default'},{k:v for k,v in data[1].items() if k!='default'})
                            else:self.assertEqual(got[1:],data[1:])
                        else:self.assertEqual(got,data)
        for file in ROOT.rglob('*.py'):
            if 'fixtures' in file.parts:continue
            tree=ast.parse(file.read_text())
            self.assertFalse(any(isinstance(n,ast.ImportFrom) and any(a.name=='*' for a in n.names) for n in ast.walk(tree)),str(file))
    def test_four_edge_fixes(self):
        out,bounds=run('BoundedImageCropWithMask',image=self.image,mask=self.mask,padding_left=0,padding_right=0,padding_top=0,padding_bottom=0)
        self.assertEqual(out.shape,(2,16,25,3));self.assertEqual(plain(bounds[0]),plain(bounds[1]))
        shifted=torch.zeros_like(self.mask);shifted[:,8:24,10:35]=1
        paired=torch.cat([self.mask,shifted])
        out,bounds=run('BoundedImageCropWithMask',image=self.image,mask=paired,padding_left=0,padding_right=0,padding_top=0,padding_bottom=0)
        torch.testing.assert_close(out[0],self.image[0,4:20,5:30]);torch.testing.assert_close(out[1],self.image[1,8:24,10:35])
        mismatch=torch.cat([paired,self.mask])
        out,_=run('BoundedImageCropWithMask',image=self.image,mask=mismatch,padding_left=0,padding_right=0,padding_top=0,padding_bottom=0)
        torch.testing.assert_close(out[1],self.image[1,4:20,5:30])
        with self.assertRaisesRegex(ValueError,'different sizes'):
            run('SwwanImageCropByMask',image=self.image,mask=torch.cat([self.mask,torch.zeros_like(self.mask)]))
        out=run('SwwanImageCropByMask',image=self.image,mask=torch.zeros_like(self.mask))[0]
        torch.testing.assert_close(out,self.image)
        image,mask=run('SwwanImageCropByMaskBatch',image=self.image,masks=torch.zeros_like(self.mask),width=24,height=16)
        self.assertEqual(image.shape,(0,16,24,3));self.assertEqual(mask.shape,(0,16,24))
        with self.assertRaisesRegex(ValueError,'at least one'):run('SwwanImageListToImageBatch',images=[],device=['cpu'])
    def test_concat_defaults_and_grid(self):
        a,b=self.image[:1],self.image[1:]
        base=run('SwwanImageConcanate',image1=a,image2=b,direction='right',match_image_size=False)[0]
        multi=run('SwwanImageConcatMulti',image_1=a,image_2=b,inputcount=2,direction='right')[0]
        torch.testing.assert_close(base,multi,atol=0,rtol=0)
        tiles=[a,b,1-a,1-b]
        got=run('SwwanImageConcatMulti',inputcount=4,layout='grid',columns=2,**{f'image_{i+1}':x for i,x in enumerate(tiles)})[0]
        expected=run('SwwanImageGridComposite2x2',**{f'image{i+1}':x for i,x in enumerate(tiles)})[0]
        torch.testing.assert_close(got,expected,atol=0,rtol=0)
        batch=run('SwwanImageConcatMulti',image_1=torch.cat(tiles),layout='batch_grid',columns=2)[0]
        torch.testing.assert_close(batch,expected,atol=0,rtol=0)
    def test_math_presets_and_rejection(self):
        namespace={'math':math,'ANY_TYPE':'*'}
        exec((ROOT/'tests/fixtures/baseline_calculate.py').read_text(),namespace)
        calculate=namespace['math_calculate']();mathnode=C['MathExpression_UTK']()
        presets=importlib.import_module(reg.__name__+'.ops.math_presets').PRESETS
        for expr,_ in presets:
            if expr=='custom':continue
            with self.subTest(expression=expr):
                a=2 if expr=='acosh(a)' or any(op in expr for op in ['&','|','^','<<','>>']) else .5
                old=calculate.calculate(expr,'',a,2,3)
                new=mathnode.evaluate('',a=a,b=2,c=3,preset=expr)['result']
                self.assertEqual(new,(old[1],old[0],old[2]))
        self.assertEqual(mathnode.evaluate('a.width + b',a=self.image,b=2)['result'],(42,42.,True))
        self.assertEqual(C['math_calculate']().calculate('custom',"__import__('os').system('false')",1),(0.,0,False))
        with self.assertRaises(ValueError):mathnode.evaluate('a.__class__',a=1)
    def test_aspect_exact_reference(self):
        namespace=dict(vars(importlib.import_module(reg.__name__+'.ops.aspect_resize')))
        exec((ROOT/'tests/fixtures/baseline_aspect.py').read_text(),namespace)
        reference=namespace['ImageScaleByAspectRatioV2']()
        for ratio in ['1:1','original','custom']:
            for fit in ['letterbox','crop','fill']:
                for method in ['lanczos','bicubic','hamming','bilinear','box','nearest']:
                    with self.subTest(ratio=ratio,fit=fit,method=method):
                        expected=reference.image_scale_by_aspect_ratio(ratio,3,2,fit,method,'8','longest',48,'#ffffff',self.image,self.mask)
                        got=run('ImageResizeKJv2Alternative',image=self.image,mask=self.mask,resize_mode='aspect_ratio',aspect_ratio=ratio,proportional_width=3,proportional_height=2,aspect_fit=fit,aspect_method=method,aspect_length=48,fill_color='#ffffff')
                        torch.testing.assert_close(got[0],expected[0],atol=0,rtol=0);torch.testing.assert_close(got[3],expected[1],atol=0,rtol=0)
                        self.assertEqual(got[1:3],expected[3:5])
    def test_save_files_paths_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            previous=folder_paths.get_output_directory();folder_paths.set_output_directory(directory)
            try:
                alpha=torch.linspace(0,1,40).repeat(1,32,1)
                for fmt in ['png','webp','tif','jpg','bmp']:
                    result=run('SwwanSaveImage',image=self.image[:1],alpha=alpha,file_format=fmt,output_path='nested',filename_prefix='test',webp_lossless=True,caption='caption',save_workflow_as_json=True,extra_pnginfo={'workflow':{'nodes':[]}},prompt={'1':{'class_type':'test'}})
                    path=Path(result['result'][0][0]);self.assertTrue(path.is_absolute());self.assertEqual(path.parent,(Path(directory)/'nested').resolve())
                    self.assertEqual(path.with_suffix('.txt').read_text(),'caption');self.assertEqual(json.loads(path.with_suffix('.json').read_text()),{'nodes':[]})
                    img=Image.open(path);self.assertEqual(img.size,(40,32))
                    if fmt in ['png','webp','tif']:
                        self.assertEqual(img.mode,'RGBA');np.testing.assert_array_equal(np.asarray(img)[...,3],(alpha[0].numpy()*255).astype('uint8'))
                    actual=np.asarray(img)[...,:3].astype('int16')
                    expected=(self.image[0].numpy()*255).astype('uint8')
                    if fmt in ['png','webp','tif']:np.testing.assert_array_equal(actual,expected)
                    else:
                        expected=((self.image[:1]*alpha.unsqueeze(-1)+(1-alpha.unsqueeze(-1)))[0].numpy()*255).astype('uint8')
                        if fmt=='bmp':np.testing.assert_array_equal(actual,expected)
                        else:self.assertLess(np.abs(actual-expected).mean(),5)
                    if fmt=='png':self.assertIn('workflow',img.info)
                    if fmt in ['jpg','bmp']:self.assertEqual(img.mode,'RGB')
                    img.close()
                one=run('SwwanSaveImage',image=self.image[:1],file_format='png',filename_prefix='duplicate')['result'][0][0]
                two=run('SwwanSaveImage',image=self.image[:1],file_format='png',filename_prefix='duplicate')['result'][0][0]
                self.assertNotEqual(one,two)
                sentinel=Path(directory)/'duplicate_00002.txt';sentinel.write_text('do not overwrite')
                three=run('SwwanSaveImage',image=self.image[:1],file_format='webp',filename_prefix='duplicate',caption='new')['result'][0][0]
                self.assertTrue(three.endswith('00003.webp'));self.assertEqual(sentinel.read_text(),'do not overwrite')
                directories=run('SwwanSaveImage',image=self.image,output_path=['first','second'],number_prefix=True,filename_prefix='separate')['result'][0]
                self.assertTrue(all(Path(x).name.startswith('00000_') for x in directories))
                # Premultiplied round trip through the same transparent pipeline.
                a=alpha.unsqueeze(-1);straight=self.image[:1]
                saved=run('SwwanSaveImage',image=straight*a,alpha=alpha,file_format='png',input_mode='premultiplied')['result'][0][0]
                img=Image.open(saved);np.testing.assert_array_equal(np.asarray(img)[...,3],(alpha[0].numpy()*255).astype('uint8'));img.close()
                run('SwwanSaveImageKJ',images=self.image[:1],output_folder='custom',filename_prefix='legacy')
                self.assertTrue(list((Path(directory)/'custom').glob('*.png')))
            finally:folder_paths.set_output_directory(previous)
    def test_migration_ownership_and_idempotence(self):
        api={'1':{'class_type':'ImageConcatMulti','inputs':{'image_1':['2',0]}},'2':{'class_type':'SaveImageKJ','inputs':{}},'3':{'class_type':'GetImageSize','inputs':{}}}
        untouched,report=migrate(api);self.assertEqual(untouched,api);self.assertEqual(report['ambiguous'],['1','2'])
        changed,report=migrate(api,selected=['1']);self.assertEqual(changed['1']['class_type'],'SwwanImageConcatMulti');self.assertEqual(changed['2'],api['2'])
        again,_=migrate(changed,selected=['1']);self.assertEqual(again,changed)
        invalid={'8':{'class_type':'ImageResizeKJv2Alternative','inputs':{'keep_proportion':False}}}
        corrected,_=migrate(invalid);self.assertEqual(corrected['8']['inputs']['keep_proportion'],'stretch')
        graph=json.loads((ROOT/'examples/qwen2511-remove-single-swwan.json').read_text());new,_=migrate(graph,registry=reg);again,_=migrate(new,registry=reg);self.assertEqual(new,again)
    def test_cpu_example_execution(self):
        import asyncio
        import execution
        import nodes
        asyncio.run(nodes.load_custom_node(str(opts.comfyui_root/'comfy_extras/nodes_mask.py'),module_parent='comfy_extras'))
        nodes.NODE_CLASS_MAPPINGS.update(C)
        with tempfile.TemporaryDirectory() as directory:
            old=folder_paths.get_output_directory();folder_paths.set_output_directory(directory)
            try:
                for filename,expected,alpha in [('cpu-image-tools.json',(256,256),False),('cpu-rgba.json',(128,96),True)]:
                    workflow=json.loads((ROOT/'examples'/filename).read_text());prompt={}
                    for node in workflow['nodes']:
                        cls=nodes.NODE_CLASS_MAPPINGS[node['type']]
                        inputs=dict(zip(defaults(cls),node['widgets_values']))
                        for field in node['inputs']:
                            if field.get('link') is not None:
                                link=next(x for x in workflow['links'] if x[0]==field['link']);inputs[field['name']]=[str(link[1]),link[2]]
                        prompt[str(node['id'])]={'class_type':node['type'],'inputs':inputs}
                    engine=execution.PromptExecutor(__import__('server').PromptServer.instance,cache_type=execution.CacheType.CLASSIC,cache_args={'lru':0,'ram':0,'ram_inactive':0})
                    engine.execute(prompt,filename,{'extra_pnginfo':{'workflow':workflow}},[str(workflow['last_node_id'])])
                    self.assertTrue(engine.success,engine.status_messages)
                    path=sorted((Path(directory)/'swwan_examples').glob(('rgba' if alpha else 'image_tools')+'*.png'))[-1]
                    with Image.open(path) as image:
                        self.assertEqual(image.size,expected);self.assertEqual(image.mode,'RGBA' if alpha else 'RGB')
                        self.assertIn('workflow',image.info)
                        if alpha:self.assertTrue((np.asarray(image)[...,3]==127).all())
                        else:self.assertGreater(np.asarray(image).min(),0)
            finally:folder_paths.set_output_directory(old)

    def test_seed_api_metadata(self):
        prompt={'7':{'inputs':{'seed':-1}}};workflow={'nodes':[{'id':7,'widgets_values':[-1]}]}
        actual=run('SwwanSeed',seed=-1,prompt=prompt,unique_id='7',extra_pnginfo={'workflow':workflow})[0]
        self.assertGreater(actual,0);self.assertEqual(prompt['7']['inputs']['seed'],actual);self.assertEqual(workflow['nodes'][0]['widgets_values'],[actual])
        self.assertEqual(run('SwwanSeed',seed=123)[0],123)

if __name__=='__main__':unittest.main(argv=[sys.argv[0],*rest])
