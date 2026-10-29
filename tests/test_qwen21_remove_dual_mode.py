"""Exercise the shipped graph with actual Swwan CPU nodes and lazy selection."""
import argparse, importlib.util, json, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from migrate_qwen2511_workflow import load_nodes, defaults
parser = argparse.ArgumentParser(); parser.add_argument('--comfyui-root', type=Path, required=True)
opts, rest = parser.parse_known_args()
registry = load_nodes(opts.comfyui_root)
import torch
C = registry.NODE_CLASS_MAPPINGS
P = json.loads((ROOT/'examples/qwen21-remove-turbo-bluebox-gh-dual-mode-api.json').read_text())
BASE = json.loads((ROOT/'examples/qwen21-remove-turbo-bluebox-gh-expanded.json').read_text())
BASE_CROP = next(n for n in BASE['nodes'] if n['id']==6)['widgets_values_named']
spec=importlib.util.spec_from_file_location('gh_dual_reference',ROOT/'tests/fixtures/reference/gh_crop_restore_color_8d9499.py')
GH=importlib.util.module_from_spec(spec);spec.loader.exec_module(GH)


def call(kind, **kwargs):
    cls=C[kind];values={} if kind=='AnySwitch (Swwan)' else defaults(cls);values.update(kwargs)
    return getattr(cls(), cls.FUNCTION)(**values)


class Graph:
    def __init__(self, image, mask, protection, enhanced=False, block_size=32):
        self.cache={'1':(image,), '2':(mask,), '4':(protection,), '100':(enhanced,)}
        self.executed=[]; self.block_size=block_size
    def value(self, value):
        return self.run(value[0])[value[1]] if isinstance(value,list) and len(value)==2 and isinstance(value[0],str) else value
    def run(self, key):
        if key in self.cache:return self.cache[key]
        entry=P[key];kind=entry['class_type'];inputs=entry['inputs']
        if kind=='AnyBooleanSwitch (Swwan)':
            flag=self.value(inputs['开关']);obj=C[kind]()
            needed=obj.check_lazy_status(开关=flag)
            result=obj.process(开关=flag,输入=self.value(inputs['输入']) if '输入' in needed else None)
        elif kind.startswith('Primitive'):
            result=(inputs['value'],)
        else:
            args={name:self.value(v) for name,v in inputs.items()}
            if kind=='SwwanBlockifyMask':args['block_size']=self.block_size
            result=call(kind,**args)
        self.executed.append(key);self.cache[key]=result;return result


class DualMode(unittest.TestCase):
    def setUp(self):
        h,w=97,161;y,x=torch.meshgrid(torch.linspace(.1,.7,h),torch.linspace(.1,.7,w),indexing='ij')
        self.image=torch.stack((x,y,(x+y)/2),-1)[None]
        self.mask=torch.zeros(1,h,w);self.mask[:,10:80,10:20]=1;self.mask[:,60:80,10:105]=1;self.mask[:,40:45,95:100]=.25
        self.protect=torch.zeros_like(self.mask);self.protect[:,55:69,25:40]=1
    def graph(self, enhanced=False):return Graph(self.image,self.mask,self.protect,enhanced)
    def test_normal_preserves_pixels_softness_and_lazy_branch(self):
        g=self.graph();effective=g.run('104')[1]
        expected=call('SwwanMaskCombine',operation='相减',bbox_mode='关闭',alignment='居中',mask_1=self.mask,mask_2=self.protect)[0]
        self.assertTrue(torch.equal(effective,expected));self.assertTrue(torch.any(effective==.25))
        baseline=call('SwwanCropByMaskV5',**BASE_CROP,image=self.image,mask=expected)
        got=g.run('6');self.assertTrue(torch.equal(got[0],baseline[0]));self.assertTrue(torch.equal(got[5],baseline[5]));self.assertEqual(got[6:],baseline[6:])
        painted=call('SwwanDrawMaskOnImage',image=baseline[0],mask=baseline[5],color='0, 255, 0',opacity=.4,device='cpu')[0]
        self.assertTrue(torch.equal(g.run('7')[0],painted));self.assertEqual(g.run('112')[1],'移除图像中的绿色区域')
        self.assertFalse({'101','102','106','110'} & set(g.executed))
    def test_enhanced_expands_region_and_excludes_protection(self):
        g=self.graph(True);m=g.run('104')[1]
        self.assertTrue(torch.any((m>0)&(self.mask==0)&(self.protect==0)))
        self.assertFalse(torch.any(m[self.protect==1]));self.assertEqual(g.run('108')[1],1.)
        self.assertIn('自然重建',g.run('112')[1]);p=g.run('6');marked=g.run('7')[0]
        self.assertTrue(torch.equal(marked[p[5]==1],torch.tensor([0.,1.,0.]).expand_as(marked[p[5]==1])))
        self.assertEqual(P['6']['inputs']['mask'],P['36']['inputs']['mask'])
    def test_native_gh_and_final_gate_for_both_modes(self):
        for enhanced in (False,True):
            g=self.graph(enhanced);m=g.run('104')[1];crop=g.run('6')
            generated=(crop[0]+.06).clamp(0,1)
            restored=GH.Restore().恢复图像(crop[4],generated,4,5,crop[5],颜色还原强度=1.)[0]
            final=torch.where((m>0)[...,None],restored,self.image)
            self.assertEqual(final.shape,self.image.shape)
            self.assertTrue(torch.equal(final[m==0],self.image[m==0]))
            self.assertTrue(torch.equal(final[self.protect==1],self.image[self.protect==1]))
    def test_edge_disconnected_and_all_protected(self):
        self.mask.zero_();self.mask[:,:3,:4]=.3;self.mask[:,-5:,-6:]=1
        for enhanced in (False,True):
            g=self.graph(enhanced);self.assertEqual(g.run('6')[9:11],(161,97))
            self.assertTrue(torch.equal(g.run('104')[1][self.protect==1],torch.zeros_like(self.mask[self.protect==1])))
        for enhanced in (False,True):
            g=Graph(self.image,self.mask,torch.ones_like(self.mask),enhanced)
            with self.assertRaisesRegex(ValueError,'no selected pixels'):g.run('6')
    def test_common_control_and_shared_inference_contract(self):
        for key in ('103','107','111'):self.assertEqual(P[key]['inputs']['开关'],['100',0])
        self.assertFalse(P['100']['inputs']['value']);self.assertEqual(P['101']['inputs']['block_size'],32)
        self.assertEqual(P['113']['inputs']['mask'],['104',1]);self.assertEqual(P['14']['inputs']['resolution'],['6',8]);self.assertEqual(P['19']['inputs']['latent_image'],['14',2]);self.assertEqual(P['21']['inputs']['颜色还原强度'],1.)
        w=json.loads((ROOT/'examples/qwen21-remove-turbo-bluebox-gh-dual-mode.json').read_text());nodes={n['id']:n for n in w['nodes']}
        for lid,source,slot,target,index,kind in w['links']:
            self.assertIn(lid,nodes[source]['outputs'][slot]['links']);self.assertEqual(nodes[target]['inputs'][index]['link'],lid)
        baseline={n['id']:n for n in BASE['nodes']}
        for key in (6,9,10,11,12,13,15,16,17,18,19,20,21):self.assertEqual(nodes[key].get('widgets_values'),baseline[key].get('widgets_values'))

if __name__=='__main__':unittest.main(argv=[sys.argv[0],*rest])
