"""Recording CPU encoders verify routing, image order and strict Qwen 2.1 contracts."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'pruna_encoding_tests'
for name, path in ((PACKAGE, ROOT), (PACKAGE + '.ops', ROOT / 'ops'), (PACKAGE + '.nodes', ROOT / 'nodes')):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = load(PACKAGE + '.ops.qwen21_pruna_encoding', ROOT / 'ops/qwen21_pruna_encoding.py')
N = load(PACKAGE + '.nodes.qwen21_pruna_encoding', ROOT / 'nodes/qwen21_pruna_encoding.py')


class NodeOutput:
    def __init__(self, *values):
        self.result = values


class Official:
    calls = []
    wrong_size = False

    @classmethod
    def define_schema(cls):
        return None

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images=None):
        cls.calls.append(dict(clip=clip, prompt=prompt, negative_prompt=negative_prompt, vae=vae,
                              resolution=resolution, images=images))
        height, width = next(iter(images.values())).shape[1:3] if images else (1024, 1024)
        if cls.wrong_size:
            width += 32
        return NodeOutput(['positive'], ['negative'], {'samples': torch.zeros(1, 64, height // 16, width // 16)})


class ModelConfig:
    FUNCTION = 'configure_model'

    def configure_model(self, instruction=''):
        return ({'model_name': 'qwen_image21', 'vae_unit': 16, 'llama_template': instruction},)


class EditUtils:
    FUNCTION = 'encode'
    calls = []
    wrong_padding = False
    wrong_size = False

    def encode(self, clip, vae, prompt, model_config, configs):
        self.calls.append(dict(clip=clip, vae=vae, prompt=prompt, model_config=model_config, configs=configs))
        main = configs[0]['image']
        if self.wrong_size:
            main = main[:, :-32]
        h, w = main.shape[1:3]
        pad = {'x': 0, 'y': 0, 'width': int(self.wrong_padding) * 16, 'height': 0, 'scale_by': 1.}
        return ([[torch.ones(1, 2, 4), {'reference_latents': ['main']}]],
                {'samples': torch.zeros(1, 64, h // 16, w // 16)},
                {'ref_latents': [object() for _ in configs]}, main, None, pad)


class ZeroOut:
    FUNCTION = 'zero_out'

    def zero_out(self, conditioning):
        return ([[torch.zeros_like(t[0]), dict(t[1])] for t in conditioning],)


class Encoding(unittest.TestCase):
    def setUp(self):
        Official.calls = []
        Official.wrong_size = False
        EditUtils.calls = []
        EditUtils.wrong_padding = False
        EditUtils.wrong_size = False
        self.registry = {'TextEncodeQwenImage21': Official, 'QwenImage21ModelConfig_EditUtils': ModelConfig,
                         'EditTextEncode_EditUtils': EditUtils, 'ConditioningZeroOut': ZeroOut}
        self.image = torch.zeros(1, 64, 96, 3)
        self.config = dict(task='edit', encoder='auto', prompt='edit the object', reference_resolution=128)

    def encode(self, config=None, image=None, references=None, registry=None):
        return E.encode(config or self.config, 'clip', 'vae', self.image if image is None else image,
                        references, self.registry if registry is None else registry)

    def test_official_preserves_main_size_order_and_negative(self):
        references = [torch.full((1, 35, 51, 3), index / 20) for index in range(1, 12)]
        config = dict(self.config, negative_prompt='no artifacts')
        positive, negative, latent, record = self.encode(config, references=references)
        call = Official.calls[-1]
        self.assertEqual(call['resolution'], 0)
        self.assertEqual(call['negative_prompt'], 'no artifacts')
        self.assertEqual(list(call['images']), [f'image_{i}' for i in range(1, 13)])
        self.assertIs(call['images']['image_1'], self.image)
        for index, reference in enumerate(references, 2):
            prepared = call['images'][f'image_{index}']
            self.assertTrue(torch.equal(prepared[:, :35, :51], reference))
            self.assertEqual(prepared.shape[1:3], (64, 64))
        self.assertEqual(latent['samples'].shape, (1, 64, 4, 6))
        self.assertEqual(record['canvas_size'], (96, 64))
        self.assertEqual(record['backend'], 'official')
        self.assertEqual((positive, negative), (['positive'], ['negative']))

    def test_reference_budget_and_rgba_are_independent(self):
        reference = torch.ones(1, 400, 800, 4)
        reference[..., 3] = .5
        prepared = E.prepare_references([reference], 128)[0]
        self.assertEqual(prepared.shape, (1, 96, 192, 4))
        torch.testing.assert_close(prepared[..., 3], torch.full_like(prepared[..., 3], .5))
        self.assertEqual(reference.shape, (1, 400, 800, 4))

    def test_text_mode_never_encodes_images_and_uses_qwen_latent(self):
        config = dict(self.config, task='photoreal', generation_mode='text', output_width=768, output_height=512)
        _, _, latent, record = self.encode(config, image='unused', references='unused')
        self.assertEqual(Official.calls[-1]['images'], {})
        self.assertEqual(latent['samples'].shape, (1, 64, 32, 48))
        self.assertFalse(bool(torch.any(latent['samples'])))
        self.assertTrue(record['text_only'])
        self.assertEqual(record['image_sizes'], [])
        self.assertEqual(N.Qwen21PrunaEncode().check_lazy_status(config, image=None, references=None), [])

    def test_text_mode_does_not_silently_replace_selected_encoder(self):
        config = dict(self.config, task='photoreal', generation_mode='text', encoder='editutils')
        with self.assertRaisesRegex(ValueError, 'requires the official'):
            self.encode(config)
        self.assertFalse(Official.calls or EditUtils.calls)
        # A remembered text toggle must not remove image conditioning from other tasks.
        self.encode(dict(self.config, generation_mode='text'))
        self.assertIn('image_1', Official.calls[-1]['images'])

    def test_cutout_preserves_requested_flags_and_reports_promotion(self):
        config = dict(self.config, task='cutout')
        _, negative, _, record = self.encode(config)
        call = EditUtils.calls[-1]
        item = call['configs'][0]
        self.assertFalse(item['to_ref'])
        self.assertTrue(item['to_vl'])
        self.assertTrue(item['ref_main_image'])
        self.assertEqual(item['ref_longest_edge'], 96)
        self.assertEqual(item['ref_resize_mode'], 'longest_edge')
        self.assertEqual(item['ref_crop'], 'pad')
        self.assertTrue(torch.equal(negative[0][0], torch.zeros_like(negative[0][0])))
        self.assertTrue(record['cutout_reference_promoted'])
        self.assertEqual(record['requested_to_ref'], [False])
        self.assertEqual(record['actual_reference_count'], 1)
        self.assertFalse(Official.calls)

    def test_bfs_keeps_main_and_reference_roles(self):
        reference = torch.ones(1, 64, 32, 3)
        _, _, _, record = self.encode(dict(self.config, task='face_swap'), references=[reference])
        configs = EditUtils.calls[-1]['configs']
        self.assertEqual([item['ref_main_image'] for item in configs], [True, False])
        self.assertEqual([item['to_ref'] for item in configs], [True, True])
        self.assertEqual([item['ref_longest_edge'] for item in configs], [96, 64])
        self.assertIs(configs[0]['image'], self.image)
        self.assertTrue(torch.equal(configs[1]['image'], reference))
        self.assertEqual(record['actual_reference_count'], 2)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.encode(dict(self.config, task='face_swap'), references=[])

    def test_missing_backend_never_falls_back(self):
        for config, registry, missing in [
            (self.config, {}, 'TextEncodeQwenImage21'),
            (dict(self.config, task='cutout'), {'TextEncodeQwenImage21': Official}, 'QwenImage21ModelConfig_EditUtils'),
        ]:
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, missing):
                self.encode(config, registry=registry)
        self.assertFalse(Official.calls)

    def test_second_geometry_transform_is_rejected(self):
        Official.wrong_size = True
        with self.assertRaisesRegex(RuntimeError, 'latent shape'):
            self.encode()
        config = dict(self.config, task='cutout')
        EditUtils.wrong_size = True
        with self.assertRaisesRegex(RuntimeError, 'second geometry'):
            self.encode(config)
        EditUtils.wrong_size = False
        EditUtils.wrong_padding = True
        with self.assertRaisesRegex(RuntimeError, 'unexpected padding'):
            self.encode(config)

    def test_invalid_inputs_and_unconnected_lazy_inputs(self):
        for image in (torch.zeros(1, 63, 96, 3), torch.zeros(2, 64, 96, 3), torch.full((1, 64, 96, 3), float('nan'))):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.encode(image=image)
        with self.assertRaisesRegex(ValueError, 'only its marked'):
            self.encode(dict(self.config, task='cutout'), references=[self.image])
        with self.assertRaisesRegex(ValueError, 'clear negative_prompt'):
            self.encode(dict(self.config, task='cutout', negative_prompt='no noise'))
        node = N.Qwen21PrunaEncode()
        self.assertEqual(node.check_lazy_status(self.config), [])
        self.assertEqual(node.check_lazy_status(self.config, image=None), ['image'])
        self.assertEqual(node.check_lazy_status(self.config, image=self.image, references=None), ['references'])

    def test_unpad_keeps_native_alpha_and_checks_canvas(self):
        _, _, _, record = self.encode(dict(self.config, task='cutout'))
        result = torch.rand(1, 64, 96, 4)
        self.assertIs(E.unpad(result, record), result)
        with self.assertRaisesRegex(ValueError, 'Decoded canvas'):
            E.unpad(torch.zeros(1, 32, 96, 4), record)
        with self.assertRaisesRegex(ValueError, 'same Pruna encode'):
            E.unpad(result, {})


if __name__ == '__main__':
    unittest.main()
