"""Registered crop-node forwarding, defaults and output compatibility on CPU."""
import argparse
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from migrate_qwen2511_workflow import defaults, load_nodes

parser = argparse.ArgumentParser()
parser.add_argument('--comfyui-root', type=Path, required=True)
options, remaining = parser.parse_known_args()
registry = load_nodes(options.comfyui_root)
import torch

Crop = registry.NODE_CLASS_MAPPINGS['SwwanCropByMaskV5']


class CropBudget(unittest.TestCase):
    def run_crop(self, **overrides):
        args = defaults(Crop)
        args.pop('qwen_min_resolution')
        args.pop('qwen_max_resolution')
        image = torch.linspace(0, 1, 73 * 101 * 3).reshape(1, 73, 101, 3)
        mask = torch.zeros(1, 73, 101)
        mask[:, :12, :20] = .6
        args.update(image=image, mask=mask, crop_mode='qwen21_aligned',
                    qwen_resolution_mode='crop_area', edit_expansion_mode='factor')
        args.update(overrides)
        return Crop().crop_by_mask_v5(**args)

    def test_optional_input_appended_and_new_defaults(self):
        schema = Crop.INPUT_TYPES()['optional']
        self.assertEqual(list(schema)[-2:], ['qwen_max_resolution', 'qwen_min_resolution'])
        self.assertEqual(schema['qwen_min_resolution'][1]['default'], 128)
        self.assertEqual(schema['qwen_max_resolution'][1]['default'], 1024)
        signature = inspect.signature(Crop.crop_by_mask_v5)
        self.assertEqual(signature.parameters['qwen_min_resolution'].default, 128)
        self.assertEqual(signature.parameters['qwen_max_resolution'].default, 1024)
        old = self.run_crop()
        explicit = self.run_crop(qwen_min_resolution=128, qwen_max_resolution=1024)
        self.assertEqual(len(old), 12)
        self.assertEqual(old[8], 0)
        self.assertEqual(old[11], 128)
        for index in (0, 1, 3, 5):
            self.assertTrue(torch.equal(old[index], explicit[index]))
        self.assertEqual(old[2], explicit[2])

    def test_saved_explicit_bounds_still_override_defaults(self):
        out = self.run_crop(qwen_min_resolution=512, qwen_max_resolution=1920)
        self.assertEqual(out[11], 512)
        self.assertEqual(out[4]['max_resolution'], 1920)

    def test_ui_help_covers_all_inputs_and_outputs(self):
        for group in Crop.INPUT_TYPES().values():
            for name, data in group.items():
                with self.subTest(input=name):
                    self.assertGreater(len(data[1]['tooltip']), 20)
                    self.assertTrue(any('\u4e00' <= c <= '\u9fff' for c in data[1]['tooltip']))
        self.assertEqual(len(Crop.OUTPUT_TOOLTIPS), len(Crop.RETURN_TYPES))
        self.assertTrue(all(len(tip) > 20 for tip in Crop.OUTPUT_TOOLTIPS))
        for text in ('bounds', 'edit_region', 'qwen21_aligned', 'crop_area', 'fixed',
                     'qwen_min_resolution=128', 'qwen_max_resolution=1024',
                     '面积预算', '不动点', 'SEAM', '颜色校正', '首张', '空选区', '零位移'):
            self.assertIn(text, Crop.DESCRIPTION)

    def test_minimum_reaches_both_restore_backends(self):
        for backend in ('swwan', 'goohai'):
            with self.subTest(backend=backend):
                out = self.run_crop(qwen_min_resolution=257, qwen_restore_backend=backend)
                self.assertEqual(out[11], 288)
                self.assertEqual(tuple(out[0].shape[1:3]), (out[7], out[6]))
                self.assertEqual(out[6] % 32, 0)
                self.assertEqual(out[7] % 32, 0)
                seam = out[4] if backend == 'swwan' else out[4]['swwan_preprocess']
                self.assertEqual(seam['min_resolution'], 257)
                self.assertEqual(seam['resolution'], 288)

    def test_conflicting_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, 'minimum exceeds maximum'):
            self.run_crop(qwen_min_resolution=769, qwen_max_resolution=799)

    def test_low_budget_thin_selection_reaches_both_backends(self):
        image = torch.zeros(1, 128, 500, 3)
        mask = torch.zeros(1, 128, 500)
        mask[:, 2:3, 50:450] = 1
        for backend in ('swwan', 'goohai'):
            with self.subTest(backend=backend):
                out = self.run_crop(image=image, mask=mask,
                                    edit_expansion_mode='reserve_ratio', qwen_restore_backend=backend)
                self.assertEqual(out[9:], (402, 3, 128))
                self.assertEqual((out[6], out[7]), (256, 64))
                self.assertGreater(float(out[5].sum()), 0)
                self.assertTrue(torch.equal(out[1][..., 0], out[5]))

    def test_fixed_and_edit_region_do_not_apply_dynamic_minimum(self):
        out = self.run_crop(qwen_resolution_mode='fixed', qwen_resolution=512,
                            qwen_min_resolution=1024, qwen_max_resolution=768)
        self.assertEqual(out[11], 512)
        base = self.run_crop(crop_mode='edit_region')
        other = self.run_crop(crop_mode='edit_region', qwen_min_resolution=1024,
                              qwen_max_resolution=768)
        self.assertTrue(torch.equal(base[0], other[0]))
        self.assertEqual(base[2], other[2])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
