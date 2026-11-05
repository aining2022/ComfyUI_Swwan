"""Pruna BFS graph contracts and CPU recovery; no detector or model download."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--comfyui-root', type=Path, required=True)
args, remaining = parser.parse_known_args()
package = types.ModuleType('pruna_face_tests')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for sub in ('ops', 'nodes'):
    module = types.ModuleType(package.__name__ + '.' + sub)
    module.__path__ = [str(ROOT / sub)]
    sys.modules[module.__name__] = module
spec = importlib.util.spec_from_file_location('comfy_execution.graph_utils', args.comfyui_root / 'comfy_execution/graph_utils.py')
parent = types.ModuleType('comfy_execution')
parent.__path__ = []
sys.modules[parent.__name__] = parent
graphs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = graphs
spec.loader.exec_module(graphs)
from pruna_face_tests.ops import qwen21_pruna_face as F
from pruna_face_tests.ops import qwen21_geometry as G
from pruna_face_tests.ops.qwen21_graph import row_order
from pruna_face_tests.nodes import qwen21_pruna_face as N


def config(**values):
    return dict(dict(task='face_swap', people=1, swap_head=False, forehead=False,
                     skin_match=False, seed=42, prompt='head_swap', qwen_min_resolution=128,
                     qwen_max_resolution=1024, expand_percent=5, feather_percent=5), **values)


class FaceGraphs(unittest.TestCase):
    def check_links(self, graph):
        for key, entry in graph.items():
            for name, value in entry['inputs'].items():
                if graphs.is_link(value):
                    self.assertIn(value[0], graph, f'{key}.{name}: {value}')

    def test_root_is_detection_only_and_no_old_config_consumers(self):
        out = F.build_face_graph('model', 'clip', 'vae', 'image', 'mask', 'protection',
                                 ['ref'], config(), manual=True)
        self.check_links(out['expand'])
        kinds = {n['class_type'] for n in out['expand'].values()}
        self.assertIn('SwwanQwen21PrunaFaceSequence', kinds)
        self.assertIn('SwwanQwen21PrunaFaceGuard', kinds)
        self.assertNotIn('SwwanQwen21Core', kinds)
        self.assertNotIn('SamplerCustomAdvanced', kinds)
        for n in out['expand'].values():
            if 'config' in n['inputs']:
                self.assertTrue(n['class_type'].startswith('SwwanQwen21Pruna'))

    def test_six_people_share_model_and_keep_standard_pruna_sampling(self):
        out = F.build_face_sequence('patched', 'clip', 'vae', 'original', 'mask', 'protection',
                                    ((1024, 1024), [object()] * 6), True, 6,
                                    ['ref'] * 6, config(people=6))
        graph = out['expand']
        self.check_links(graph)
        kinds = [n['class_type'] for n in graph.values()]
        self.assertEqual(kinds.count('SamplerCustomAdvanced'), 6)
        self.assertEqual(kinds.count('SwwanQwen21PrunaEncode'), 6)
        self.assertEqual(kinds.count('GuHai_FaceAlignment'), 6)
        self.assertEqual(kinds.count('RMBG'), 6)
        self.assertEqual(kinds.count('SwwanColorMatch'), 6)
        self.assertFalse(any('Loop' in k or 'Lora' in k or 'Sage' in k or k == 'SwwanQwen21Core' for k in kinds))
        self.assertEqual([n['inputs']['index'] for n in graph.values()
                          if n['class_type'] == 'SwwanQwen21ReferenceAtIndex'], list(range(6)))
        for n in graph.values():
            kind, inputs = n['class_type'], n['inputs']
            if kind == 'ManualSigmas':
                self.assertEqual(inputs['sigmas'], F.PRUNA_SIGMAS)
            if kind == 'CFGGuider':
                self.assertEqual(inputs['model'], 'patched')
                self.assertEqual(inputs['cfg'], 1.)
            if kind == 'SwwanCropByMaskV5' and inputs['crop_mode'] == 'qwen21_aligned':
                self.assertEqual(inputs['qwen_resolution_mode'], 'crop_area')
                self.assertEqual(inputs['qwen_min_resolution'], 128)
                self.assertEqual(inputs['qwen_max_resolution'], 1024)
                self.assertFalse(inputs['fill_mask_holes'])
                self.assertEqual(graph[inputs['mask'][0]]['class_type'], 'SwwanQwen21PrunaFaceMask')
        restores = [n['inputs'] for n in graph.values() if n['class_type'] == 'SwwanQwen21PrunaFaceRestore']
        self.assertEqual(restores[0]['background'], 'original')
        self.assertEqual(restores[0]['previous_report'], '')
        for n in restores[1:]:
            self.assertEqual(graph[n['background'][0]]['class_type'], 'SwwanQwen21PrunaFaceRestore')
            self.assertEqual(n['background'][0], n['previous_report'][0])
            self.assertEqual(n['previous_report'][1], 1)
        self.assertTrue(all(n['original'] == 'original' for n in restores))

    def test_row_order_stays_frozen(self):
        refs = [types.SimpleNamespace(bbox=(x, y, x+30, y+50), label=(x, y))
                for x, y in ((100, 200), (0, 200), (100, 3), (0, 0))]
        ordered = row_order(((300, 300), refs))[1]
        self.assertEqual([s.label for s in ordered], [(0, 0), (100, 3), (0, 200), (100, 200)])
        self.assertEqual(refs[0].label, (100, 200))

    def test_inputs_fail_before_sampling(self):
        image = torch.zeros(1, 41, 67, 3)
        mask = torch.zeros(1, 41, 67)
        mask[:, 5:15, 5:15] = 1
        with self.assertRaisesRegex(ValueError, 'task=face_swap'):
            N.Qwen21PrunaFace().run(None, None, None, image, [image], config(task='remove'))
        with self.assertRaisesRegex(ValueError, 'reference'):
            N.Qwen21PrunaFace().run(None, None, None, image, [], config())
        with self.assertRaisesRegex(ValueError, 'fully protected'):
            N.Qwen21PrunaFace().run(None, None, None, image, [image], config(), mask, mask)
        with self.assertRaisesRegex(ValueError, 'same dimensions'):
            F.source_masks(image, torch.zeros(1, 4, 4))
        with self.assertRaisesRegex(ValueError, 'require 2 references'):
            F.build_face_sequence(None, None, None, image, mask, mask, ((41, 67), [1, 2]),
                                  True, 2, [image], config(people=2))
        with self.assertRaisesRegex(ValueError, 'no editable pixels'):
            F.guarded_selection(image, mask, mask)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            F.source_masks(torch.full_like(image, float('nan')))
        with self.assertRaisesRegex(ValueError, 'skin_strength'):
            F.validate_config(config(skin_strength=1.1))


class FaceRestoration(unittest.TestCase):
    def test_manual_and_protected_pixels_remain_original(self):
        image = torch.linspace(0, .7, 81*131*3).reshape(1, 81, 131, 3)
        selection = torch.zeros(1, 81, 131)
        selection[:, 15:60, 20:95] = 1
        protection = torch.zeros_like(selection)
        protection[:, 22:32, 40:50] = 1
        effective = F.guarded_selection(image, selection, protection)
        p, pm, seam = G.prepare_aligned(image, effective, resolution_mode='crop_area')
        background = image.clone()
        background[protection == 1] = .9
        out, report = F.restore_person(background, image, torch.ones_like(p), seam, pm, True,
                                       selection, protection, config(), 0)
        self.assertTrue(torch.equal(out[selection == 0], image[selection == 0]))
        self.assertTrue(torch.equal(out[protection == 1], image[protection == 1]))
        self.assertTrue(bool(torch.any(out[effective > 0] != image[effective > 0])))
        final, text = F.finish(out, image, protection, report, 1, config())
        parsed = json.loads(text)
        self.assertEqual(parsed['people_processed'], 1)
        self.assertEqual(parsed['sampling'], 'pruna_v0.2_6step')
        self.assertEqual(parsed['protection_changed_values'], 0)
        self.assertTrue(torch.equal(final, out))

    def test_failure_does_not_claim_completed_people(self):
        image = torch.zeros(1, 51, 71, 3)
        mask = torch.ones(1, 51, 71)
        p, pm, seam = G.prepare_aligned(image, mask, resolution_mode='crop_area')
        protection = torch.zeros_like(mask)
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            F.restore_person(image, image, p[:, :-1], seam, pm, True, mask, protection, config(), 0)
        with self.assertRaisesRegex(ValueError, 'order/report'):
            F.restore_person(image, image, p, seam, pm, True, mask, protection, config(), 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            F.finish(image, image, protection, '{"people_records":[]}', 1, config())
        with self.assertRaisesRegex(ValueError, 'dimensions/channels'):
            F.finish(image[:, :1], image, protection, '{"people_records":[]}', 1, config())


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
