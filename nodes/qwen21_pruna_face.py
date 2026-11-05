# SPDX-License-Identifier: MIT
"""Inspectable BFS expansion using Pruna sampling and original-canvas protection."""


class Qwen21PrunaFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',),
                             'image': ('IMAGE',), 'references': ('QWEN21_REFS',),
                             'config': ('QWEN21_PRUNA_CONFIG',)},
                'optional': {'mask': ('MASK',), 'protect_mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'report')
    FUNCTION = 'run'
    DESCRIPTION = 'BFS换脸/换头专用分支。保留区域检测、参考脸对齐、RMBG及可选肤色匹配；按从上到下、同行从左到右匹配1–6张参考。共享外部已打补丁的Pruna模型，每人6步串行生成。可选mask为手动选区，空mask走检测；protect_mask白色区域最终保留原图。不是GPU常驻适配器实现。'

    def run(self, model, clip, vae, image, references, config, mask=None, protect_mask=None):
        from ..ops.qwen21_pruna_face import validate_config, source_masks, build_face_graph
        validate_config(config)
        if not references:
            raise ValueError('BFS requires at least one reference image')
        selected, protection, manual = source_masks(image, mask, protect_mask)
        return build_face_graph(model, clip, vae, image, selected, protection, references, config, manual)


class Qwen21PrunaFaceGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'count': ('INT', {'default': 1}),
                             'valid': ('BOOLEAN', {'default': False}), 'references': ('QWEN21_REFS',),
                             'config': ('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'

    def run(self, image, count, valid, references, config):
        from ..ops.qwen21_pruna_face import validate_config
        validate_config(config)
        if not valid or not 1 <= count <= min(6, config.get('people', 1)):
            raise ValueError('No valid face/head regions; no sampling started')
        if len(references) < count:
            raise ValueError(f'{count} people require {count} references; received {len(references)}')
        return (image,)


class Qwen21PrunaFaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'mask': ('MASK',), 'protect_mask': ('MASK',)}}
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'run'

    def run(self, image, mask, protect_mask):
        from ..ops.qwen21_pruna_face import guarded_selection
        return (guarded_selection(image, mask, protect_mask),)


class Qwen21PrunaFaceReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}
    RETURN_TYPES = ('QWEN21_REFS',)
    FUNCTION = 'run'

    def run(self, image):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('Each BFS reference must contain exactly one image')
        return ([image],)


class Qwen21PrunaFaceSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',),
                             'image': ('IMAGE',), 'mask': ('MASK',), 'protect_mask': ('MASK',),
                             'regions': ('SEGS',), 'manual': ('BOOLEAN', {'default': False}),
                             'count': ('INT', {'default': 1}), 'references': ('QWEN21_REFS',),
                             'config': ('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    FUNCTION = 'run'

    def run(self, model, clip, vae, image, mask, protect_mask, regions, manual, count, references, config):
        from ..ops.qwen21_pruna_face import build_face_sequence
        return build_face_sequence(model, clip, vae, image, mask, protect_mask,
                                   regions, manual, count, references, config)


class Qwen21PrunaFaceRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'background': ('IMAGE',), 'original': ('IMAGE',), 'image': ('IMAGE',),
                             'seam': ('SEAM',), 'mask': ('MASK',), 'selection': ('MASK',),
                             'protect_mask': ('MASK',), 'manual': ('BOOLEAN', {'default': False}),
                             'config': ('QWEN21_PRUNA_CONFIG',), 'person': ('INT', {'default': 0}),
                             'previous_report': ('STRING', {'default': ''})}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    FUNCTION = 'run'

    def run(self, **kwargs):
        from ..ops.qwen21_pruna_face import restore_person
        return restore_person(**kwargs)


class Qwen21PrunaFaceFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'original': ('IMAGE',), 'protect_mask': ('MASK',),
                             'report': ('STRING', {'default': ''}), 'count': ('INT', {'default': 1}),
                             'config': ('QWEN21_PRUNA_CONFIG',)}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    FUNCTION = 'run'

    def run(self, **kwargs):
        from ..ops.qwen21_pruna_face import finish
        return finish(**kwargs)
