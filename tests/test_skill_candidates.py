"""Four decision scenarios use the live generated catalog and skill, not a copied list."""
import importlib.util
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'skills/comfyui-swwan-node-development/scripts/find_candidates.py'
spec=importlib.util.spec_from_file_location('candidates',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows={r['id']:r for r in json.loads((ROOT/'docs/node-catalog.json').read_text())}
class SkillScenarios(unittest.TestCase):
    def test_duplicate_resize_reuses_existing(self):
        matches=module.candidates('缩放 resize',ROOT,8)
        self.assertIn('ImageResizeKJv2Alternative',{r['id'] for r in matches})
        self.assertIn('resize_mode',rows['ImageResizeKJv2Alternative']['schema']['optional'])
    def test_batch_bug_optimizes_existing(self):
        matches=module.candidates('批次 batch list',ROOT,20)
        self.assertIn('SwwanImageListToImageBatch',{r['id'] for r in matches})
        self.assertEqual(rows['SwwanImageListToImageBatch']['output_types'],['IMAGE'])
        # An empty-list defect is a behavior correction, not another interface.
        self.assertIn('at least one image',(ROOT/'image_batch_utils.py').read_text())
    def test_independent_mask_contract_allows_new(self):
        match=next(r for r in module.candidates('遮罩 blockify',ROOT,30) if r['id']=='SwwanBlockifyMask')
        self.assertEqual(match['output_types'],['MASK'])
        self.assertIn('block_size',match['schema']['required'])
        self.assertNotEqual(match['output_types'],rows['SwwanCropByMaskV5']['output_types'])
    def test_third_party_port_requires_provenance(self):
        resize=rows['SwwanImageResizeRange']
        self.assertIn('GPL',resize['license']);self.assertIn('图像缩放范围孤海',(ROOT/'scripts/migrate_qwen2511_workflow.py').read_text())
        self.assertNotIn('图像缩放范围孤海',rows)
        text=(ROOT/'skills/comfyui-swwan-node-development/SKILL.md').read_text()
        for word in ['完全覆盖','输出顺序','第三方实现','许可','node_manifest.json','动态端口']:self.assertIn(word,text)
if __name__=='__main__':unittest.main()
