# SPDX-License-Identifier: MIT
"""Portable graph and atomic geometry contracts, without model/GPU execution."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_qwen21_pruna_expanded import TASKS, api_from_ui, build_task, validate

package = types.ModuleType('expanded_geometry_tests')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for name in ('ops', 'nodes'):
    module = types.ModuleType(package.__name__ + '.' + name)
    module.__path__ = [str(ROOT / name)]
    sys.modules[module.__name__] = module
G = importlib.import_module(package.__name__ + '.ops.qwen21_geometry')
N = importlib.import_module(package.__name__ + '.nodes.qwen_geometry')


class Geometry(unittest.TestCase):
    def test_odd_edge_padded_geometry_rgb_rgba(self):
        for channels in (3, 4):
            for rect in ((0, 0, 11, 81), (93, 53, 101, 83), (31, 23, 63, 54)):
                image = torch.full((1, 83, 101, channels), .5)
                mask = torch.zeros(1, 83, 101)
                x, y, r, b = rect
                mask[:, y:b, x:r] = 1
                prepared, _, seam = G.prepare_aligned(image, mask, resolution_mode='crop_area')
                patch, x, y, w, h, ow, oh = N.QwenGeometryUnpack().unpack(prepared, seam)
                self.assertEqual((ow, oh), (101, 83))
                self.assertEqual(patch.shape, (1, h, w, channels))
                self.assertEqual((x, y, x+w, y+h), seam['box'])
                self.assertTrue(torch.allclose(patch, torch.full_like(patch, .5), atol=1e-5))

    def test_transparent_canvas_roundtrip_keeps_completion_outside_selection(self):
        image = torch.zeros(1, 83, 101, 3)
        mask = torch.zeros(1, 83, 101)
        mask[:, 25:55, 35:65] = 1
        prepared, _, seam = G.prepare_aligned(image, mask, resolution_mode='crop_area')
        generated = torch.full((*prepared.shape[:3], 4), .5)
        patch, x, y, w, h, ow, oh = N.QwenGeometryUnpack().unpack(generated, seam, True)
        rgb = torch.zeros(1, oh, ow, 3)
        inverse_alpha = torch.ones(1, oh, ow)
        rgb[:, y:y+h, x:x+w] = patch[..., :3]
        inverse_alpha[:, y:y+h, x:x+w] *= 1-patch[..., 3]
        rgba = torch.cat((rgb, (1-inverse_alpha)[..., None]), dim=-1)
        self.assertTrue(torch.any(rgba[..., 3][mask == 0] > 0))
        self.assertTrue(torch.allclose(rgba[:, y:y+h, x:x+w, 3], torch.full((1, h, w), .5)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rgba.png'
            Image.fromarray((rgba[0].numpy()*255).round().astype(np.uint8)).save(path)
            with Image.open(path) as loaded:
                self.assertEqual((loaded.mode, loaded.size), ('RGBA', (101, 83)))
                self.assertEqual(int(np.asarray(loaded)[y, x, 3]), 128)

    def test_reject_wrong_seam_size_channels_and_rgb_cutout(self):
        image = torch.ones(1, 83, 101, 3)
        prepared, _, seam = G.prepare_aligned(image, torch.ones(1,83,101), resolution=128)
        node = N.QwenGeometryUnpack()
        for bad in (None, {}, {'qwen21_geometry_version': 2}, {'swwan_preprocess': seam}):
            with self.assertRaisesRegex(ValueError, 'Swwan'):
                node.unpack(prepared, bad)
        with self.assertRaisesRegex(ValueError, 'canvas'):
            node.unpack(prepared[:, :-1], seam)
        with self.assertRaisesRegex(ValueError, 'RGBA'):
            node.unpack(prepared, seam, True)
        with self.assertRaisesRegex(ValueError, 'BHWC'):
            node.unpack(prepared[..., :2], seam)


class GraphContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / 'examples/qwen21-pruna-v5-expanded'
        cls.schemas = json.loads((cls.folder / 'schemas.json').read_text())
        cls.source = json.loads((ROOT / 'examples/qwen21-pruna-v4-baseline.json').read_text())

    def graph(self, task):
        return json.loads((self.folder / f'pruna-v5-{task}.json').read_text())

    def test_reproducible_ui_and_api(self):
        for task in TASKS:
            with self.subTest(task=task):
                w = self.graph(task)
                self.assertEqual(w, build_task(self.schemas, self.source, task))
                api = json.loads((self.folder / f'pruna-v5-{task}-api.json').read_text())
                self.assertEqual(api, api_from_ui(w))
                self.assertFalse(w.get('definitions'))
                validate(w, self.schemas)

    def test_no_node_overlap_and_explicit_contracts(self):
        for task in TASKS:
            w = self.graph(task)
            for i, node in enumerate(w['nodes']):
                x, y = node['pos']; width, height = node['size']
                for other in w['nodes'][i+1:]:
                    ox, oy = other['pos']; ow, oh = other['size']
                    self.assertFalse(x < ox+ow and ox < x+width and y < oy+oh and oy < y+height,
                                     f'{task}: overlap {node["id"]}/{other["id"]}')
                self.assertFalse(node.get('flags', {}).get('collapsed'))
                if node['type'] == 'SwwanCropByMaskV5':
                    p = node['widgets_values_named']
                    self.assertEqual((p['qwen_resolution_mode'], p['qwen_min_resolution'], p['qwen_max_resolution']),
                                     ('crop_area', 128, 1024))
                if node['type'] == 'SwwanQwenGeometryUnpack':
                    self.assertTrue(node['widgets_values_named']['require_alpha'])
                if node['type'] == 'SwwanLoadAndResizeImage':
                    self.assertFalse(node['widgets_values_named']['resize'])
                    self.assertEqual(node['widgets_values_named']['background_color'], '')
            sigmas = next(n for n in w['nodes'] if n['type'] == 'ManualSigmas')
            self.assertEqual(sigmas['widgets_values_named']['sigmas'],
                             '1.0, 0.961054, 0.918949, 0.769302, 0.545554, 0.222292, 0')

    def test_all_overview_only_edit_is_enabled(self):
        w = json.loads((self.folder / 'pruna-v5-expanded-all.json').read_text())
        validate(w, self.schemas)
        self.assertEqual(len(w['groups']), 9)
        edit_ids = {n['id'] for n in self.graph('edit')['nodes']}
        for node in w['nodes']:
            if node['id'] not in edit_ids:
                self.assertEqual(node['mode'], 2)

    def test_reject_broken_links_and_hidden_wrapper(self):
        w = self.graph('edit')
        broken = copy.deepcopy(w)
        node = next(n for n in broken['nodes'] if n['id'] == broken['links'][0][3])
        node['inputs'][broken['links'][0][4]]['link'] = None
        with self.assertRaisesRegex(ValueError, 'bidirectional'):
            validate(broken, self.schemas)
        broken = copy.deepcopy(w)
        broken['nodes'][0]['type'] = 'SwwanQwen21PrunaInfer'
        with self.assertRaisesRegex(ValueError, 'Hidden'):
            validate(broken, self.schemas)

    def test_geometry_schema_matches_actual_class(self):
        actual = json.loads(json.dumps(N.QwenGeometryUnpack.INPUT_TYPES()))
        self.assertEqual(actual, self.schemas['SwwanQwenGeometryUnpack']['input'])
        manifest = json.loads((ROOT/'node_manifest.json').read_text())
        self.assertEqual(len([n for n in manifest if n['id']=='SwwanQwenGeometryUnpack']), 1)


if __name__ == '__main__':
    unittest.main()
