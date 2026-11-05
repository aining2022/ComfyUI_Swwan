# SPDX-License-Identifier: MIT
"""Pruna CPU task, protection, loader and model-routing contracts; no GPU inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--comfyui-root', type=Path, required=True)
options, remaining = parser.parse_known_args()
package = types.ModuleType('pruna_task_tests')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
for name in ('ops', 'nodes'):
    module = types.ModuleType(package.__name__ + '.' + name)
    module.__path__ = [str(ROOT / name)]
    sys.modules[module.__name__] = module
spec = importlib.util.spec_from_file_location(
    'comfy_execution.graph_utils', options.comfyui_root / 'comfy_execution/graph_utils.py')
module = types.ModuleType('comfy_execution')
module.__path__ = []
sys.modules[module.__name__] = module
graphs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = graphs
spec.loader.exec_module(graphs)
from pruna_task_tests.ops import qwen21_pruna as P
from pruna_task_tests.nodes import qwen21_pruna as N


def config(**changes):
    values = {}
    for group in ('required', 'optional'):
        for key, declaration in N.Qwen21PrunaControls.INPUT_TYPES().get(group, {}).items():
            kind = declaration[0]
            metadata = declaration[1] if len(declaration) > 1 else {}
            values[key] = metadata.get('default', kind[0] if isinstance(kind, list) else None)
    values.update(extra_prompt='test edit', color_mode='none', feather_percent=0)
    values.update(changes)
    return P.resolve(values)


def fixture(channels=3):
    image = torch.linspace(.05, .7, 61*97*channels).reshape(1, 61, 97, channels)
    mask = torch.zeros(1, 61, 97)
    mask[:, 17:43, 29:62] = 1
    protect = torch.zeros_like(mask)
    protect[:, 23:31, 38:46] = 1
    return image, mask, protect


class TaskContracts(unittest.TestCase):
    def test_task_prompts_strengths_and_encoder_routing(self):
        for task in P.TASKS:
            c = config(task=task, extra_prompt='custom instruction')
            self.assertEqual(c['task_lora'], P.LORAS[task])
            self.assertEqual(c['active_lora_strength'], P.STRENGTHS[task])
            self.assertTrue(c['prompt'].endswith('custom instruction'))
            self.assertEqual(c['active_additional_strength'], .6 if task == 'edit' else 0)
        self.assertIn('head_swap', config(task='face_swap')['prompt'])
        self.assertIn('<mva>', config(task='angles')['prompt'])
        self.assertIn('绿色', config(task='remove')['prompt'])
        self.assertTrue(P.is_text(config(task='photoreal', generation_mode='text')))
        self.assertFalse(P.uses_references(config(task='photoreal', generation_mode='text')))
        self.assertFalse(P.is_text(config(task='edit', generation_mode='text')))
        with self.assertRaises(ValueError):
            config(extra_prompt='')
        with self.assertRaises(ValueError):
            config(qwen_min_resolution=1024, qwen_max_resolution=512)

    def test_local_crop_budget_and_exact_protection(self):
        image, mask, protection = fixture(4)
        c = config(edit_scope='local')
        clean, selected, seam, report = P.prepare(image, mask, protection, c)
        self.assertEqual(seam['resolution'], 128)
        self.assertEqual(seam['resolution_mode'], 'crop_area')
        self.assertTrue(all(x % 32 == 0 for x in seam['canvas_size']))
        self.assertLess(clean.shape[1]*clean.shape[2], 1024*1024)
        self.assertEqual(json.loads(report)['working_resolution'], 128)
        result, fusion, report = P.restore(torch.ones_like(clean), seam, c)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(torch.equal(result[mask == 0], image[mask == 0]))
        self.assertTrue(torch.equal(result[protection == 1], image[protection == 1]))
        self.assertTrue(torch.equal(result[..., 3], image[..., 3]))
        self.assertTrue(torch.all(fusion[protection == 1] == 0))
        self.assertEqual(json.loads(report)['outside_changed_values'], 0)
        self.assertTrue(bool(torch.any(result[..., :3][(mask-protection) > 0] != image[..., :3][(mask-protection) > 0])))

    def test_full_transforms_do_not_feather_back_original_borders(self):
        image, _, _ = fixture()
        for task in ('edit', 'photoreal', 'angles', 'watermark'):
            c = config(task=task, edit_scope='full', feather_percent=30)
            clean, _, seam, _ = P.prepare(image, None, None, c)
            result, fusion, _ = P.restore(torch.ones_like(clean), seam, c)
            torch.testing.assert_close(result, torch.ones_like(image))
            torch.testing.assert_close(fusion, torch.ones_like(fusion))

    def test_full_image_geometry_does_not_shrink_to_protection(self):
        image, mask, protection = fixture()
        c = config(edit_scope='full')
        clean, _, seam, _ = P.prepare(image, mask, protection, c)
        self.assertEqual(seam['native_crop_size'], (97, 61))
        result, _, _ = P.restore(torch.ones_like(clean), seam, c)
        self.assertTrue(torch.equal(result[protection == 1], image[protection == 1]))
        self.assertGreater(float((result-image).abs().sum()), 0)
        self.assertEqual(P.color_mode(config(color_mode='auto'), seam), 'none')
        self.assertEqual(P.color_mode(config(edit_scope='local', color_mode='auto'), seam), 'goohai')

    def test_invalid_inputs_stop_before_sampling(self):
        image, mask, _ = fixture()
        c = config(task='remove')
        for selected, protected in ((None, None), (mask*0, None), (mask, mask)):
            with self.assertRaises(ValueError):
                P.prepare(image, selected, protected, c)
        with self.assertRaises(ValueError):
            P.prepare(image.repeat(2, 1, 1, 1), mask, None, c)
        with self.assertRaises(ValueError):
            P.prepare(image, mask[:, :-1], None, c)
        with self.assertRaisesRegex(ValueError, 'BFS'):
            P.prepare(image, mask, None, config(task='face_swap'))
        with self.assertRaisesRegex(ValueError, 'protection'):
            P.prepare(image, mask, mask, config(task='cutout'))

    def test_brush_marker_retains_opacity_and_alpha(self):
        image = torch.full((1, 11, 13, 4), .5)
        mask = torch.zeros(1, 11, 13)
        mask[:, 2:9, 3:10] = .5
        original = image.clone()
        marked = P.mark(image, mask, config(task='remove'))
        self.assertTrue(torch.equal(image, original))
        self.assertTrue(torch.equal(marked[..., 3], image[..., 3]))
        torch.testing.assert_close(marked[0, 5, 5, :3], torch.tensor([.4, .6, .4]))
        self.assertTrue(torch.equal(marked[mask == 0], image[mask == 0]))
        enhanced = P.mark(image, mask, config(task='remove', enhanced=True))
        torch.testing.assert_close(enhanced[0, 5, 5, :3], torch.tensor([.25, .75, .25]))
        self.assertIs(P.mark(image, mask, config(task='edit')), image)

    def test_circle_selection_closes_region_and_rechecks_protection(self):
        image, mask, protection = fixture()
        circle = mask.clone()
        circle[:, 18:42, 30:61] = 0
        selected = P.selection(image, circle, protection, config(task='remove', marking='circle'))
        self.assertEqual(float(selected[0, 35, 50]), 1)
        self.assertTrue(torch.all(selected[protection == 1] == 0))
        circle[:, 17, 40] = 0
        with self.assertRaises(ValueError):
            P.selection(image, circle, protection, config(task='remove', marking='circle'))

    def test_enhanced_mask_rechecks_protection_after_block_expansion(self):
        image, mask, protection = fixture()
        calls = []
        class Block:
            def process(self, selected, size, device):
                calls.append((size, device))
                return (torch.ones_like(selected),)
        stub = types.ModuleType('pruna_task_tests.workflow_tools')
        stub.BlockifyMask = Block
        with patch.dict(sys.modules, {stub.__name__: stub}):
            selected = P.selection(image, mask, protection, config(task='remove', enhanced=True))
        self.assertEqual(calls, [(32, 'cpu')])
        self.assertTrue(torch.all(selected[protection == 1] == 0))
        self.assertTrue(torch.all(selected[protection == 0] == 1))

    def test_outpaint_uses_gray_and_strict_original_canvas(self):
        image, _, protection = fixture(4)
        c = config(task='outpaint', left=11, right=7, top=5, bottom=0, seam_band=0)
        clean, _, seam, _ = P.prepare(image, None, protection, c)
        source = seam['original_image']
        self.assertEqual(source.shape[1:3], (66, 115))
        self.assertTrue(torch.equal(source[:, 5:66, 11:108], image))
        self.assertTrue(torch.all(source[:, :5, :, :3] == .5))
        self.assertTrue(torch.all(source[:, :5, :, 3] == 1))
        result, _, _ = P.restore(torch.ones_like(clean), seam, c)
        self.assertTrue(torch.equal(result[:, 5:66, 11:108], image))
        self.assertGreater(float(result[:, :4].mean()), .5)
        band = config(task='outpaint', left=11, right=0, top=0, bottom=0, seam_band=2)
        _, _, seam, _ = P.prepare(image, None, protection, band)
        self.assertTrue(torch.all(seam['original_mask'][:, :, 11:13] == 1))
        self.assertTrue(torch.all(seam['original_mask'][:, :, 13:] == 0))
        with self.assertRaises(ValueError):
            P.prepare(image, None, None, config(task='outpaint', left=0, right=0, top=0, bottom=0))

    def test_cutout_requires_native_rgba_and_does_not_clip_to_selection(self):
        image, mask, _ = fixture()
        c = config(task='cutout', context_factor=2.)
        clean, selected, seam, _ = P.prepare(image, mask, None, c)
        marked = P.mark(clean, selected, c)
        self.assertTrue(torch.any(marked != clean))
        with self.assertRaisesRegex(ValueError, 'RGBA'):
            P.restore(torch.ones_like(clean), seam, c)
        generated = torch.cat((torch.ones_like(clean), torch.full_like(clean[..., :1], .6)), -1)
        result, alpha, _ = P.restore(generated, seam, c)
        self.assertEqual(result.shape, (1, 61, 97, 4))
        self.assertTrue(torch.equal(alpha, result[..., 3]))
        self.assertTrue(bool(torch.any(alpha[mask == 0] > 0)))
        self.assertLess(float(alpha.max()), .61)

    def test_color_backends_receive_clean_geometry_and_final_gate(self):
        image, mask, protection = fixture(4)
        for mode in ('goohai', 'bluebox'):
            with self.subTest(mode=mode):
                c = config(task='remove', color_mode=mode)
                clean, _, seam, _ = P.prepare(image, mask, protection, c)
                calls = []
                class GH:
                    FUNCTION = 'run'
                    def run(self, metadata, generated, expand, feather, selected, **kwargs):
                        calls.append((metadata, generated, kwargs))
                        return (torch.ones_like(image[..., :3]),)
                class Blue:
                    FUNCTION = 'run'
                    def run(self, reference, generated, **kwargs):
                        calls.append((reference, generated, kwargs))
                        return {'result': (torch.ones_like(generated), None, 0.)}
                stub = types.ModuleType('nodes')
                stub.NODE_CLASS_MAPPINGS = {'GH_CropRestore': GH, 'ColorCalibratorBlueBox': Blue}
                with patch.dict(sys.modules, {'nodes': stub}):
                    result, _, report = P.restore(torch.ones_like(clean), seam, c)
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(result[mask == 0], image[mask == 0]))
                self.assertTrue(torch.equal(result[protection == 1], image[protection == 1]))
                self.assertTrue(torch.equal(result[..., 3], image[..., 3]))
                self.assertEqual(json.loads(report)['color_backend'], mode)
                if mode == 'goohai':
                    self.assertTrue(torch.equal(calls[0][0]['原始图像'], image[..., :3]))
                else:
                    self.assertTrue(torch.equal(calls[0][0], clean[..., :3]))
                    self.assertIs(calls[0][2]['mask'], seam['prepared_mask'])
        stub.NODE_CLASS_MAPPINGS = {}
        with patch.dict(sys.modules, {'nodes': stub}), self.assertRaises(RuntimeError):
            P.restore(clean, seam, c)

    def test_frozen_native_gh_restore_rgb_rgba_local_edge_and_outpaint(self):
        spec = importlib.util.spec_from_file_location(
            'pruna_native_gh_reference', ROOT / 'tests/fixtures/reference/gh_crop_restore_color_8d9499.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        registry = types.ModuleType('nodes')
        registry.NODE_CLASS_MAPPINGS = {'GH_CropRestore': native.Restore}
        with patch.dict(sys.modules, {'nodes': registry}):
            for channels in (3, 4):
                for mode in ('local', 'edge', 'outpaint'):
                    with self.subTest(channels=channels, mode=mode):
                        image, selected, protection = fixture(channels)
                        values = dict(task='edit', edit_scope='local', color_mode='goohai',
                                      color_strength=1., expand_percent=4, feather_percent=5)
                        if mode == 'edge':
                            selected.zero_()
                            selected[:, :8, :35] = 1
                            protection.zero_()
                            protection[:, 2:5, 4:10] = 1
                        elif mode == 'outpaint':
                            values.update(task='outpaint', left=11, right=7, top=5, bottom=0, seam_band=0)
                        c = config(**values)
                        clean, _, seam, _ = P.prepare(image, selected, protection, c)
                        generated = (clean[..., :3] + .04).clamp(0, 1)
                        result, fusion, report = P.restore(generated, seam, c)
                        background = seam['original_image']
                        allowed = seam['original_mask']
                        protected = seam['protection']
                        expected_size = (1, 66, 115, channels) if mode == 'outpaint' else image.shape
                        self.assertEqual(result.shape, expected_size)
                        self.assertTrue(bool(torch.isfinite(result).all()))
                        self.assertTrue(torch.equal(result[allowed == 0], background[allowed == 0]))
                        self.assertTrue(torch.equal(result[protected == 1], background[protected == 1]))
                        self.assertTrue(torch.all(fusion[protected == 1] == 0))
                        self.assertEqual(json.loads(report)['outside_changed_values'], 0)
                        if channels == 4:
                            self.assertTrue(torch.equal(result[..., 3], background[..., 3]))
                        if mode == 'outpaint':
                            self.assertTrue(torch.equal(result[:, 5:66, 11:108], image))
                            if channels == 4:
                                self.assertTrue(torch.all(result[:, :5, :, 3] == 1))
                        self.assertGreater(float((result-background).abs().sum()), 0)

    def test_text_mode_does_not_consume_images_or_references(self):
        c = config(task='photoreal', generation_mode='text', output_width=160, output_height=96)
        self.assertEqual(N.Qwen21PrunaInputs().check_lazy_status(c, image=None, mask=None, references=None), [])
        with patch.dict(sys.modules, {'nodes': types.ModuleType('nodes')}):
            image, mask, protection, refs = N.Qwen21PrunaInputs().load(
                c, image_filename='does-not-exist.png', reference_filenames='missing.png')
        self.assertEqual(refs, [])
        clean, _, seam, _ = P.prepare(object(), object(), object(), c)
        self.assertEqual(clean.shape, (1, 96, 160, 3))
        restored, _, _ = P.restore(clean, seam, c)
        self.assertIs(restored, clean)
        with self.assertRaises(ValueError):
            P.restore(clean[:, :-1], seam, c)

    def test_external_inputs_override_filenames_and_unused_sources(self):
        image, mask, protection = fixture()
        c = config(edit_scope='local')
        refs = [image]
        with patch.dict(sys.modules, {'nodes': types.ModuleType('nodes')}):
            result = N.Qwen21PrunaInputs().load(c, 'missing', 'missing', 'missing', 'missing',
                                                image, mask, protection, refs)
        for actual, expected in zip(result, (image, mask, protection, refs)):
            self.assertIs(actual, expected)
        active = N.Qwen21PrunaInputs().check_lazy_status(c, image=None, mask=None, protect_mask=None, references=None)
        self.assertEqual(set(active), {'image', 'mask', 'protect_mask', 'references'})
        with patch.dict(sys.modules, {'nodes': types.ModuleType('nodes')}):
            result = N.Qwen21PrunaInputs().load(config(task='watermark'), mask_filename='missing',
                reference_filenames='missing', image=image, references=refs)
        self.assertEqual(result[3], [])
        self.assertTrue(torch.all(result[1] == 0))

    def test_input_fingerprint_handles_unresolved_config_and_lazy_files(self):
        folders = types.ModuleType('folder_paths')
        with tempfile.TemporaryDirectory() as directory:
            folders.get_annotated_filepath = lambda name: str(Path(directory) / name)
            image = Path(directory) / 'input.bin'
            image.write_bytes(b'first')
            c = config(edit_scope='full')
            prompt = {'0': {'class_type': 'SwwanQwen21PrunaControls', 'inputs': c},
                      '1': {'class_type': 'SwwanQwen21PrunaInputs', 'inputs': {'config': ['0', 0]}}}
            with patch.dict(sys.modules, {'folder_paths': folders}):
                kwargs = dict(config=None, prompt=prompt, unique_id='1', image_filename='input.bin',
                              mask_filename='missing-mask.bin', reference_filenames='')
                first = N.Qwen21PrunaInputs.IS_CHANGED(**kwargs)
                image.write_bytes(b'second')
                second = N.Qwen21PrunaInputs.IS_CHANGED(**kwargs)
                self.assertNotEqual(first, second)
                prompt['1']['inputs']['image'] = ['other', 0]
                external = N.Qwen21PrunaInputs.IS_CHANGED(**kwargs, image=None)
                image.write_bytes(b'third')
                self.assertEqual(external, N.Qwen21PrunaInputs.IS_CHANGED(**kwargs, image=None))
                prompt['0']['inputs'] = config(task='photoreal', generation_mode='text')
                self.assertEqual(N.Qwen21PrunaInputs.IS_CHANGED(**kwargs), 'text')
                self.assertIsInstance(N.Qwen21PrunaInputs.IS_CHANGED(config=None, image_filename='missing'), str)

    def test_file_inputs_restore_alpha_and_load_masks_from_red(self):
        image, mask, _ = fixture()
        calls = []
        class ImageLoader:
            def load_image(self, name):
                calls.append(('image', name))
                return image, torch.full_like(mask, .3)
        class MaskLoader:
            FUNCTION = "load_image_mask"
            def load_image(self, name):
                raise AssertionError("Do not call inherited LoadImage implementation for mask files")
            def load_image_mask(self, name, channel):
                calls.append(('mask', name, channel))
                return (mask,)
        stub = types.ModuleType('nodes')
        stub.LoadImage, stub.LoadImageMask = ImageLoader, MaskLoader
        with patch.dict(sys.modules, {'nodes': stub}):
            result = N.Qwen21PrunaInputs().load(config(edit_scope='local'),
                'main.png', 'selection.png', 'protect.png', 'reference1.png\nreference2.png')
        self.assertEqual(result[0].shape[-1], 4)
        torch.testing.assert_close(result[0][..., 3], torch.full_like(mask, .7))
        self.assertEqual([call for call in calls if call[0] == 'mask'],
                         [('mask', 'selection.png', 'red'), ('mask', 'protect.png', 'red')])
        self.assertEqual(len(result[3]), 2)

    def test_task_model_starts_from_clean_base_on_every_switch(self):
        model = N.Qwen21PrunaModel()
        first = model.apply('clean_pruna', config(task='remove'))
        nodes = list(first['expand'].values())
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]['class_type'], 'LoraLoaderModelOnly')
        self.assertEqual(nodes[0]['inputs']['model'], 'clean_pruna')
        self.assertEqual(nodes[0]['inputs']['lora_name'], P.LORAS['remove'])
        second = model.apply('clean_pruna', config(task='cutout', additional_lora_mode='all'))
        nodes = list(second['expand'].values())
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]['inputs']['model'], 'clean_pruna')
        self.assertEqual(nodes[1]['inputs']['lora_name'], P.LORAS['cutout'])
        self.assertEqual(nodes[1]['inputs']['model'][0], next(iter(second['expand'])))
        self.assertEqual(model.apply('clean_pruna', config(task='remove', lora_strength=0)), ('clean_pruna',))
        third = model.apply('clean_pruna', config(task='remove'))
        self.assertEqual(next(iter(third['expand'].values()))['inputs'], next(iter(first['expand'].values()))['inputs'])

    def test_output_only_requests_selected_branch(self):
        output = N.Qwen21PrunaOutput()
        self.assertEqual(output.check_lazy_status(config()), ['generic'])
        self.assertEqual(output.check_lazy_status(config(task='face_swap')), ['face'])
        self.assertEqual(output.choose(config(), generic='ready', face='inactive'), ('ready',))
        self.assertEqual(output.choose(config(task='face_swap'), generic='inactive', face='ready'), ('ready',))
        with self.assertRaises(ValueError):
            output.choose(config())


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
