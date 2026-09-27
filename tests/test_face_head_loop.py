"""Actual EasyUse graph expansion with synthetic crop edits and final PNG save."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from migrate_qwen2511_workflow import load_nodes, defaults
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True)
args,rest=p.parse_known_args();registry=load_nodes(args.comfyui_root)
import torch
import numpy as np
from PIL import Image
import nodes,execution,folder_paths
from server import PromptServer
spec=importlib.util.spec_from_file_location('easy_loop_reference',ROOT/'tests/fixtures/easy_loop.py')
loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)


class SyntheticImage:
    @classmethod
    def INPUT_TYPES(cls): return {'required':{}}
    RETURN_TYPES=('IMAGE',);FUNCTION='make'
    def make(self): return (torch.zeros(1,24,32,3),)


class EditAndRestore:
    calls=[]
    @classmethod
    def INPUT_TYPES(cls): return {'required':{'image':('IMAGE',),'index':('INT',)}}
    RETURN_TYPES=('IMAGE',);FUNCTION='edit'
    def edit(self,image,index):
        self.calls.append(index)
        mask=torch.zeros(1,24,32);mask[:,4:12,4+index*8:10+index*8]=1
        def run(name,**kwargs):
            cls=registry.NODE_CLASS_MAPPINGS[name];v=defaults(cls);v.update(kwargs)
            return getattr(cls(),cls.FUNCTION)(**v)
        crop=run('SwwanCropByMaskV5',image=image,mask=mask,crop_mode='edit_region',alignment=0,
                 top_reserve_ratio=0,bottom_reserve_ratio=0,left_reserve_ratio=0,right_reserve_ratio=0)
        edited=torch.ones_like(crop[0])*(index+1)/4
        return (run('SwwanRestoreCropBoxV4',background_image=image,croped_image=edited,croped_mask=crop[5],crop_box=crop[2],region_info=crop[4],expand_percent=0,feather_percent=0)[0],)


class LoopRegression(unittest.TestCase):
    def test_three_iterations_without_display_states(self):
        references={'easy forLoopStart':loop.forLoopStart,'easy forLoopEnd':loop.forLoopEnd,
                    'easy whileLoopStart':loop.whileLoopStart,'easy whileLoopEnd':loop.whileLoopEnd,
                    'easy mathInt':loop.mathIntOperation,'easy compare':loop.Compare,
                    'TestImage':SyntheticImage,'TestEdit':EditAndRestore}
        previous={k:nodes.NODE_CLASS_MAPPINGS.get(k) for k in references}
        nodes.NODE_CLASS_MAPPINGS.update(references)
        previous_output=folder_paths.get_output_directory()
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder_paths.set_output_directory(directory);EditAndRestore.calls=[]
                prompt={
                    '1':{'class_type':'TestImage','inputs':{}},
                    '2':{'class_type':'easy forLoopStart','inputs':{'total':3,'initial_value1':['1',0]}},
                    '3':{'class_type':'TestEdit','inputs':{'image':['2',2],'index':['2',1]}},
                    '4':{'class_type':'easy forLoopEnd','inputs':{'flow':['2',0],'initial_value1':['3',0]}},
                    '5':{'class_type':'SaveImage','inputs':{'images':['4',0],'filename_prefix':'loop_qa'}}}
                executor=execution.PromptExecutor(PromptServer.instance,cache_type=execution.CacheType.CLASSIC,cache_args={'lru':0,'ram':0,'ram_inactive':0})
                executor.execute(prompt,'loop-repair-qa',{},['5'])
                self.assertTrue(executor.success,executor.status_messages)
                self.assertEqual(EditAndRestore.calls,[0,1,2])
                files=list(Path(directory).glob('loop_qa*.png'));self.assertEqual(len(files),1)
                out=np.asarray(Image.open(files[0]));self.assertEqual(out.shape,(24,32,3))
                for index in range(3):
                    self.assertEqual(int(out[8,6+index*8,0]),int((index+1)/4*255))
                self.assertEqual(int(out[0,0,0]),0)
        finally:
            folder_paths.set_output_directory(previous_output)
            for key,value in previous.items():
                if value is None: nodes.NODE_CLASS_MAPPINGS.pop(key,None)
                else: nodes.NODE_CLASS_MAPPINGS[key]=value

if __name__=='__main__':unittest.main(argv=[__file__]+rest)
