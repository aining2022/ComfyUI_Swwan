"""CPU regressions for sub-512 selection coverage; no model inference."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = load('coverage_geometry', ROOT / 'ops/qwen21_geometry.py')
S = load('coverage_seam', ROOT / 'ops/qwen21_gh_seam.py')
GH = load('coverage_gh', ROOT / 'tests/fixtures/reference/gh_crop_restore_color_8d9499.py')


class MaskCoverage(unittest.TestCase):
    def test_thin_edge_selection_and_restore(self):
        for vertical in (False, True):
            for value in (1., .6):
                with self.subTest(vertical=vertical, value=value):
                    mask = torch.zeros(1, 128, 500)
                    mask[:, 2:3, 50:450] = value
                    if vertical:
                        mask = mask.transpose(1, 2)
                    image = torch.zeros(*mask.shape, 3)
                    prepared, selected, seam = G.prepare_aligned(
                        image, mask, factors=(1.5,) * 4, resolution_mode='crop_area')
                    self.assertEqual(seam['resolution'], 128)
                    self.assertEqual(seam['native_crop_size'], (3, 402) if vertical else (402, 3))
                    self.assertTrue(bool(torch.any(selected > 0)))
                    self.assertTrue(bool(torch.any((selected > 0) & (selected < value))))
                    self.assertLessEqual(float(selected.max()), value)
                    generated = torch.ones_like(prepared)
                    restored, fusion = G.restore_aligned(image, generated, seam, cropped_mask=selected)
                    self.assertTrue(bool(torch.all(fusion[mask > 0] > 0)))
                    self.assertTrue(torch.equal(restored[mask == 0], image[mask == 0]))
                    gh = GH.Restore().恢复图像(S.to_goohai_seam(seam), generated, 0, 0,
                                             selected, 颜色还原强度=0)[0]
                    gated = torch.where((mask > 0)[..., None], gh, image)
                    self.assertTrue(bool(torch.all(gated[mask > 0] > 0)))
                    self.assertTrue(torch.equal(gated[mask == 0], image[mask == 0]))

    def test_disconnected_pixels_keep_both_components(self):
        mask = torch.zeros(1, 16, 16)
        mask[:, 2, 2] = .6
        mask[:, 14, 14] = .3
        selected = G.resize_mask(mask, 4, 4, preserve_coverage=True)
        expected = torch.zeros(1, 4, 4)
        expected[:, 0, 0] = .6 / 16
        expected[:, 3, 3] = .3 / 16
        torch.testing.assert_close(selected, expected)

    def test_mixed_scale_keeps_soft_coverage(self):
        mask = torch.zeros(1, 16, 4)
        mask[:, 2, 1] = .6
        selected = G.resize_mask(mask, 8, 4, preserve_coverage=True)
        expected = F.interpolate(F.interpolate(mask[:, None], size=(4, 4), mode='area'),
                                 size=(4, 8), mode='bilinear', align_corners=False)[:, 0]
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        self.assertGreater(float(selected.sum()), 0)
        self.assertLess(float(selected.max()), .6)

    def test_legacy_mask_resize_and_budget_unchanged(self):
        for value in (1., .6):
            mask = torch.zeros(1, 128, 500)
            mask[:, 2:3, 50:450] = value
            expected = F.interpolate(mask[:, None], size=(64, 256),
                                     **({'mode': 'nearest'} if value == 1 else
                                        {'mode': 'bilinear', 'align_corners': False}))[:, 0]
            torch.testing.assert_close(G.resize_mask(mask, 256, 64), expected, rtol=0, atol=0)
            with patch.object(G, 'resize_mask', wraps=G.resize_mask) as resize:
                _, selected, seam = G.prepare_aligned(
                    torch.zeros(1, 128, 500, 3), mask, factors=(1.5,) * 4,
                    resolution_mode='crop_area', min_resolution=512)
            self.assertEqual(seam['resolution'], 512)
            self.assertEqual(seam['canvas_size'], (1376, 192))
            self.assertFalse(resize.call_args.kwargs['preserve_coverage'])
            self.assertGreater(float(selected.sum()), 0)

    def test_empty_resized_mask_stops_before_inference(self):
        image = torch.zeros(1, 128, 500, 3)
        mask = torch.zeros(1, 128, 500)
        mask[:, 2:3, 50:450] = 1
        def empty(mask, width, height, **kwargs):
            return mask.new_zeros((1, height, width))
        with patch.object(G, 'resize_mask', side_effect=empty):
            with self.assertRaisesRegex(ValueError, 'Selection mask vanished'):
                G.prepare_aligned(image, mask, resolution_mode='crop_area')


if __name__ == '__main__':
    unittest.main()
