"""Validate expanded graphs against ComfyUI's actual GraphBuilder without starting a GPU."""
import argparse,importlib.util,sys,types,unittest
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfyui-root',type=Path,required=True);args,rest=p.parse_known_args()
spec=importlib.util.spec_from_file_location('comfy_execution.graph_utils',args.comfyui_root/'comfy_execution/graph_utils.py')
parent=types.ModuleType('comfy_execution');parent.__path__=[];sys.modules['comfy_execution']=parent
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from test_qwen21_unified import B,T
class Graphs(unittest.TestCase):
 def check_links(self,graph):
  for key,node in graph.items():
   for name,value in node['inputs'].items():
    if module.is_link(value):self.assertIn(value[0],graph,f'{key}.{name} -> {value}')
 def test_root_and_sequential_six(self):
  c=T.resolve('face_swap',people=6)
  root=B.build_face_graph('model','clip','vae','image','mask',['reference']*6,c)
  self.check_links(root['expand'])
  self.assertTrue(any(n['class_type']=='SwwanQwen21FaceRegions' for n in root['expand'].values()))
  seq=B.build_face_sequence('model','clip','vae','image','mask',((1024,1024),[]),False,0,6,'preview',['reference']*6,c)
  self.check_links(seq['expand'])
  kinds=[n['class_type'] for n in seq['expand'].values()]
  self.assertEqual(kinds.count('SwwanQwen21Core'),6)
  self.assertFalse(any('Loop' in k or 'UNETLoader'==k for k in kinds))
  selected=[n['inputs']['index'] for n in seq['expand'].values() if n['class_type']=='SwwanQwen21ReferenceAtIndex']
  self.assertEqual(selected,list(range(6)))
  canvases=[n['inputs']['background'] for n in seq['expand'].values() if n['class_type']=='SwwanQwen21FaceRestore']
  self.assertEqual(canvases[0],'image')
  for link in canvases[1:]:self.assertEqual(seq['expand'][link[0]]['class_type'],'SwwanQwen21FaceRestore')
if __name__=='__main__':unittest.main(argv=[sys.argv[0]]+rest)
