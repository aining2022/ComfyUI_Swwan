# SPDX-License-Identifier: MIT
"""Portable topology contracts for the fully visible Pruna BFS canvas."""
import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pruna_expanded_bfs import SIGMAS, to_api, validate_bfs


class ExpandedBFS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / 'examples/qwen21-pruna-v5-expanded'
        cls.workflow = json.loads((folder / 'pruna-v5-face_swap.json').read_text())
        cls.schemas = json.loads((folder / 'schemas.json').read_text())
        cls.original = {n['id']: n for n in json.loads(
            (ROOT / 'examples/qwen21-pruna-v4-baseline.json').read_text())['nodes']}
        cls.nodes = {n['id']: n for n in cls.workflow['nodes']}

    def incoming(self, target, name):
        index = next(i for i, n in enumerate(self.nodes[target]['inputs']) if n['name'] == name)
        return next(link[1:3] for link in self.workflow['links'] if link[3:5] == [target, index])

    def ancestors(self, target):
        result, pending = set(), [target]
        while pending:
            current = pending.pop()
            for link in self.workflow['links']:
                if link[3] == current and link[1] not in result:
                    result.add(link[1])
                    pending.append(link[1])
        return result

    def test_no_hidden_wrappers_virtual_wires_or_subgraphs(self):
        validate_bfs(self.workflow)
        self.assertFalse(self.workflow.get('definitions'))
        for node in self.nodes.values():
            self.assertFalse(node['type'].startswith('SwwanQwen21'))
            self.assertNotIn(node['type'], ('SetNode', 'GetNode'))
            self.assertFalse(node.get('flags', {}).get('collapsed', False))
            self.assertEqual(node.get('mode', 0), 0)
        self.assertIn('static', self.workflow['extra']['validation'])

    def test_original_pruna_models_and_sampling(self):
        for new_id, old_id in ((235, 509), (236, 548), (238, 511), (91001, 527),
                               (91002, 532), (91004, 526), (91005, 523), (91006, 522)):
            with self.subTest(node=new_id):
                self.assertEqual(self.nodes[new_id]['type'], self.original[old_id]['type'])
                self.assertEqual(self.nodes[new_id]['widgets_values_named'],
                                 self.original[old_id]['widgets_values_named'])
        self.assertEqual(self.nodes[91006]['widgets_values_named']['sigmas'], SIGMAS)
        self.assertEqual(self.nodes[91005]['widgets_values_named']['sampler_name'], 'euler')
        self.assertEqual(self.nodes[91004]['widgets_values_named']['cfg'], 1)
        self.assertEqual(self.nodes[91003]['widgets_values_named']['noise_seed'],
                         self.original[524]['widgets_values_named']['noise_seed'])
        self.assertEqual(self.nodes[91003]['widgets_values_named']['control_after_generate'], 'fixed')
        self.assertEqual(self.incoming(91001, 'model'), [235, 0])
        self.assertEqual(self.incoming(237, 'model'), [91001, 0])
        self.assertEqual(self.incoming(91002, 'model'), [237, 0])
        self.assertEqual(self.incoming(91004, 'model'), [91002, 0])
        for port, source in (('noise', 91003), ('guider', 91004), ('sampler', 91005), ('sigmas', 91006)):
            self.assertEqual(self.incoming(247, port), [source, 0])
        self.assertEqual(self.incoming(247, 'latent_image'), [241, 1])
        self.assertFalse(any('BlockCache' in n['type'] or 'Spectrum' in n['type']
                             for n in self.nodes.values()))

    def test_task_lora_and_trigger_visible(self):
        self.assertEqual(self.nodes[237]['widgets_values_named']['lora_name'],
                         self.original[566]['widgets_values_named']['lora_5_name'])
        self.assertEqual(self.nodes[237]['widgets_values_named']['strength_model'], 0.7)
        self.assertEqual(self.nodes[189]['widgets_values_named']['值'], 0.7)
        self.assertEqual(self.incoming(237, 'strength_model'), [189, 0])
        self.assertEqual(self.nodes[91010]['widgets_values_named'],
                         self.original[562]['widgets_values_named'])
        self.assertEqual(self.incoming(241, 'prompt'), [91010, 0])

    def test_original_detection_and_visible_accumulating_loop(self):
        for detector in (124, 233, 208):
            with self.subTest(detector=detector):
                self.assertIn(140, self.ancestors(detector))
                self.assertFalse(self.ancestors(detector) & {72, 162})
        self.assertEqual(self.nodes[72]['type'], 'easy forLoopStart')
        self.assertEqual(self.nodes[162]['type'], 'easy forLoopEnd')
        self.assertEqual(self.incoming(72, 'initial_value1'), [226, 0])
        self.assertEqual(self.incoming(226, 'in'), [140, 0])
        self.assertEqual(self.incoming(131, 'image'), [72, 2])
        self.assertEqual(self.incoming(63, 'background_image'), [72, 2])
        self.assertEqual(self.incoming(162, 'initial_value1'), [63, 0])
        self.assertEqual(self.incoming(162, 'flow'), [72, 0])
        self.assertEqual(self.incoming(110, 'segs'), [124, 0])
        self.assertEqual(self.incoming(110, 'start_index'), [72, 1])
        self.assertEqual(self.nodes[110]['widgets_values_named']['sort_rule'], '左右上下')
        self.assertEqual(self.incoming(110, 'group_threshold'), self.incoming(233, 'group_threshold'))

    def test_six_explicit_reference_slots_share_person_index(self):
        for index, loader in enumerate((200, 170, 148, 149, 150, 151)):
            with self.subTest(reference=index):
                self.assertEqual(self.nodes[loader]['type'], 'LoadImage')
                self.assertEqual(self.nodes[loader]['mode'], 0)
                self.assertEqual(self.incoming(202, f'value{index}'), [loader, 0])
        self.assertEqual(self.incoming(202, 'index'), [72, 1])

    def test_single_main_geometry_and_independent_reference_size(self):
        values = self.nodes[131]['widgets_values_named']
        self.assertEqual(values['crop_mode'], 'qwen21_aligned')
        self.assertEqual(values['qwen_resolution_mode'], 'crop_area')
        self.assertEqual((values['qwen_min_resolution'], values['qwen_max_resolution']), (128, 1024))
        self.assertEqual(values['qwen_restore_backend'], 'swwan')
        self.assertFalse(values['fill_mask_holes'])
        self.assertNotIn(138, self.nodes)
        self.assertEqual(self.incoming(63, 'crop_box'), [131, 2])
        self.assertEqual(self.incoming(63, 'region_info'), [131, 4])
        self.assertEqual(self.incoming(242, 'image'), [131, 0])
        self.assertEqual(self.incoming(282, 'image'), [131, 0])
        self.assertEqual(self.incoming(242, 'ref_longest_edge'), [282, 0])
        self.assertEqual(self.incoming(91025, 'image'), [13, 0])
        self.assertEqual(self.incoming(250, 'ref_longest_edge'), [91025, 0])
        self.assertEqual(self.nodes[249]['type'], 'CropWithPadInfo_EditUtils')
        self.assertNotIn('protect_mask', {i['name'] for i in self.nodes[63]['inputs']})

    def test_protection_is_explicit_before_crop_and_after_loop(self):
        self.assertEqual(self.nodes[284]['widgets_values_named']['value'], 0)
        self.assertEqual(self.incoming(284, 'width'), [283, 1])
        self.assertEqual(self.incoming(284, 'height'), [283, 2])
        self.assertEqual(self.nodes[91021]['widgets_values_named']['operation'], '相减')
        self.assertEqual(self.incoming(91021, 'mask_1'), [73, 0])
        self.assertEqual(self.incoming(91021, 'mask_2'), [284, 0])
        self.assertEqual(self.incoming(131, 'mask'), [91021, 0])
        self.assertEqual(self.incoming(91, 'mask'), [91021, 0])
        self.assertEqual(self.incoming(91020, 'destination'), [162, 0])
        self.assertEqual(self.incoming(91020, 'source'), [140, 0])
        self.assertEqual(self.incoming(91020, 'mask'), [284, 0])
        self.assertEqual(self.incoming(163, 'images'), [91020, 0])

    def test_api_fields_match_schemas_and_visible_wires(self):
        api = to_api(self.workflow, self.schemas)
        for key, entry in api.items():
            schema = self.schemas[entry['class_type']]['input']
            allowed = {name for group in ('required', 'optional') for name in schema.get(group, {})}
            self.assertLessEqual(set(entry['inputs']), allowed)
            self.assertFalse({'upload', 'info', 'control_after_generate'} & set(entry['inputs']))
            for socket in self.nodes[int(key)].get('inputs', []):
                if socket.get('link') is not None:
                    source, slot = self.incoming(int(key), socket['name'])
                    self.assertEqual(entry['inputs'][socket['name']], [str(source), slot])
        self.assertNotIn('91030', api)

    def test_diagnostics_and_layout_remain_inspectable(self):
        for preview, slot in ((91022, 9), (91023, 10), (91024, 11)):
            self.assertEqual(self.incoming(preview, 'source'), [131, slot])
        self.assertEqual(self.incoming(91026, 'images'), [248, 0])
        self.assertGreaterEqual(self.nodes[91030]['size'][1], 900)
        for node in self.nodes.values():
            x, y = node['pos']; width, height = node['size']
            self.assertTrue(any(gx <= x and gy <= y and x+width <= gx+gw and y+height <= gy+gh
                                for gx, gy, gw, gh in (g['bounding'] for g in self.workflow['groups'])))

    def test_validator_rejects_wrong_reciprocal_metadata(self):
        broken = copy.deepcopy(self.workflow)
        target = next(n for n in broken['nodes'] if n['id'] == 163)
        target['inputs'][0]['link'] = next(l[0] for l in broken['links'] if l[3] == 91026)
        with self.assertRaisesRegex(ValueError, 'mismatched input'):
            validate_bfs(broken)


if __name__ == '__main__':
    unittest.main()
