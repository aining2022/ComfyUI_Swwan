# SPDX-License-Identifier: MIT
"""Portable Pruna v4 topology, named API inputs and actual RGBA save tests."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_qwen21_pruna import CORE_IDS, IDS, build, to_api
from migrate_qwen2511_workflow import defaults, load_nodes
from workflow_contracts import validate_workflow

parser = argparse.ArgumentParser()
parser.add_argument('--comfyui-root', type=Path, required=True)
options, remaining = parser.parse_known_args()
registry = load_nodes(options.comfyui_root)


class PrunaWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'examples/qwen21-pruna-v4-baseline.json'
        cls.source = json.loads(cls.path.read_text())
        cls.workflow = build(cls.source, registry)
        cls.nodes = {n['id']: n for n in cls.workflow['nodes']}
        cls.api = to_api(cls.workflow)

    def incoming(self, target, name):
        socket = next(i for i, item in enumerate(self.nodes[target]['inputs']) if item['name'] == name)
        return next(link for link in self.workflow['links'] if link[3:5] == [target, socket])[1:3]

    def test_source_preserved_and_build_deterministic(self):
        before = copy.deepcopy(self.source)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(build(self.source, registry), self.workflow)
        self.assertEqual(self.source, before)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.workflow, json.loads((ROOT / 'examples/qwen21-pruna-v4.json').read_text()))
        self.assertEqual(self.api, json.loads((ROOT / 'examples/qwen21-pruna-v4-api.json').read_text()))
        validate_workflow(self.workflow, registry)

    def test_v3_models_and_sampling_preserved(self):
        original = {n['id']: n for n in self.source['nodes']}
        for node_id in CORE_IDS:
            with self.subTest(node=node_id):
                self.assertEqual(self.nodes[node_id]['type'], original[node_id]['type'])
                if node_id != 524:
                    self.assertEqual(self.nodes[node_id].get('widgets_values'), original[node_id].get('widgets_values'))
                    self.assertEqual(self.nodes[node_id].get('widgets_values_named'), original[node_id].get('widgets_values_named'))
        self.assertEqual(self.incoming(524, 'noise_seed'), [IDS['controls'], 2])
        self.assertEqual(self.nodes[524]['widgets_values_named']['control_after_generate'], 'fixed')
        self.assertEqual(self.nodes[IDS['controls']]['widgets_values_named']['seed'],
                         original[524]['widgets_values_named']['noise_seed'])
        self.assertEqual(self.incoming(525, 'latent_image'), [IDS['encode'], 2])
        self.assertEqual(self.incoming(532, 'model'), [IDS['model'], 0])
        self.assertEqual(self.incoming(IDS['model'], 'model'), [527, 0])
        self.assertEqual(self.nodes[526]['widgets_values_named']['cfg'], 1)

    def test_budget_marking_and_restore_share_record(self):
        self.assertNotIn('CutQwenImage21', {n['type'] for n in self.nodes.values()})
        self.assertNotIn('EmptyLatentImage', {n['type'] for n in self.nodes.values()})
        self.assertEqual(self.incoming(IDS['mark'], 'image'), [IDS['prepare'], 0])
        self.assertEqual(self.incoming(IDS['mark'], 'mask'), [IDS['prepare'], 1])
        self.assertEqual(self.incoming(IDS['restore'], 'seam'), [IDS['prepare'], 2])
        self.assertEqual(self.incoming(IDS['restore'], 'image'), [IDS['unpad'], 0])
        self.assertEqual(self.incoming(IDS['unpad'], 'padding'), [IDS['encode'], 3])
        controls = self.nodes[IDS['controls']]['widgets_values_named']
        self.assertEqual((controls['qwen_min_resolution'], controls['qwen_max_resolution']), (128, 1024))

    def test_lazy_output_and_unused_previews(self):
        cls = registry.NODE_CLASS_MAPPINGS['SwwanQwen21PrunaOutput']()
        self.assertEqual(cls.check_lazy_status({'task': 'edit'}), ['generic'])
        self.assertEqual(cls.check_lazy_status({'task': 'face_swap'}), ['face'])
        self.assertEqual(self.nodes[IDS['image_preview']]['mode'], 2)
        self.assertNotIn(str(IDS['image_preview']), self.api)
        self.assertEqual(self.incoming(IDS['save'], 'image'), [IDS['output'], 0])
        self.assertEqual(self.incoming(IDS['output'], 'face'), [IDS['face'], 0])
        self.assertEqual(self.incoming(IDS['output'], 'generic'), [IDS['restore'], 0])
        # Output nodes other than save may not force the generic branch or read optional files.
        for n in self.nodes.values():
            if n['type'] == 'PreviewImage': self.assertEqual(n['mode'], 2)
        self.assertFalse(any(n['type'] == 'LoadImage' for n in self.nodes.values()))

    def test_api_uses_names_and_every_link_matches(self):
        for node_id, entry in self.api.items():
            node = self.nodes[int(node_id)]
            for socket in node.get('inputs', []):
                if socket.get('link') is not None:
                    source, slot = self.incoming(node['id'], socket['name'])
                    self.assertEqual(entry['inputs'][socket['name']], [str(source), slot])
            self.assertNotIn('control_after_generate', entry['inputs'])
        inputs = self.api[str(IDS['inputs'])]['inputs']
        self.assertEqual(inputs['mask_filename'], '')
        self.assertEqual(inputs['protect_filename'], '')
        self.assertEqual(inputs['reference_filenames'], '')
        self.assertEqual(self.incoming(IDS['encode'], 'references'), [IDS['inputs'], 3])
        self.assertEqual(self.incoming(IDS['face'], 'references'), [IDS['inputs'], 3])

    def test_new_nodes_keep_complete_named_widget_contracts(self):
        for node in self.nodes.values():
            if not node['type'].startswith('Swwan'):
                continue
            with self.subTest(node=node['id']):
                schema = defaults(registry.NODE_CLASS_MAPPINGS[node['type']])
                saved = node['widgets_values_named']
                self.assertEqual(list(saved), list(schema))
                self.assertEqual(node['widgets_values'], list(saved.values()))
                for name, value in saved.items():
                    self.assertEqual(self.api[str(node['id'])]['inputs'][name], value)

    def test_tasks_preserve_v3_lora_paths_and_strength_defaults(self):
        source = {n['id']: n for n in self.source['nodes']}
        controls = self.nodes[IDS['controls']]['widgets_values_named']
        selected = source[566]['widgets_values_named']
        for index, task in enumerate(('edit', 'outpaint', 'cutout', 'watermark', 'face_swap', 'photoreal', 'remove', 'angles')):
            self.assertEqual(controls[f'task_lora_{task}'], selected[f'lora_{index+1}_name'])
        self.assertEqual(controls['additional_lora_mode'], 'edit_only')
        self.assertEqual(controls['additional_lora_strength'], .6)
        self.assertEqual(controls['extra_prompt'], source[555]['widgets_values_named']['value'])

    def test_output_saves_actual_rgba(self):
        import numpy as np
        import torch
        from PIL import Image
        cls = registry.NODE_CLASS_MAPPINGS['SwwanSaveImage']
        values = defaults(cls)
        rgba = torch.zeros(1, 9, 13, 4)
        rgba[..., 0] = .25
        rgba[:, 2:7, 3:11, 3] = .6
        values.update(self.nodes[IDS['save']]['widgets_values_named'])
        with tempfile.TemporaryDirectory() as directory:
            values.update(image=rgba, output_path=directory, embed_metadata=False, save_workflow_as_json=False)
            getattr(cls(), cls.FUNCTION)(**values)
            files = list(Path(directory).rglob('*.png'))
            self.assertEqual(len(files), 1)
            with Image.open(files[0]) as image:
                self.assertEqual(image.mode, 'RGBA')
                self.assertEqual(image.size, (13, 9))
                alpha = np.array(image)[..., 3]
                self.assertEqual(int(alpha[0, 0]), 0)
                self.assertEqual(int(alpha[4, 5]), 153)

    def test_cli_refuses_overwrite(self):
        command = [sys.executable, str(ROOT / 'scripts/build_qwen21_pruna.py'), str(self.path), str(self.path),
                   '--comfyui-root', str(options.comfyui_root)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output must be new', result.stderr)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
