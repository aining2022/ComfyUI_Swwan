# SPDX-License-Identifier: MIT
"""Visible official/EditUtils encoding contract for the shared Pruna sampler."""


class Qwen21PrunaEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'config': ('QWEN21_PRUNA_CONFIG',), 'clip': ('CLIP',), 'vae': ('VAE',)},
                'optional': {'image': ('IMAGE', {'lazy': True}), 'references': ('QWEN21_REFS', {'lazy': True})}}

    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING', 'LATENT', 'QWEN21_PRUNA_PADDING')
    RETURN_NAMES = ('positive', 'negative', 'latent', 'padding')
    FUNCTION = 'run'
    DESCRIPTION = ('自动：抠图/BFS保留EditUtils，其他任务使用官方Qwen2.1。主图须已32对齐；'
                   '官方resolution=0，EditUtils保持实际长边，发现二次缩放立即报错。'
                   '参考图按顺序处理，不套主图BOX。纯文生成仅photoreal/text，不读取任何图像。'
                   '缺少指定编码器时报错，不静默回退；当前EditUtils可能将抠图VL-only配置转成参考latent。')

    def check_lazy_status(self, config, **kwargs):
        from ..ops.qwen21_pruna_encoding import text_only
        if text_only(config):
            return []
        return [name for name in ('image', 'references') if name in kwargs and kwargs[name] is None]

    def run(self, config, clip, vae, image=None, references=None):
        from ..ops.qwen21_pruna_encoding import encode
        return encode(config, clip, vae, image, references)


class Qwen21PrunaUnpad:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'padding': ('QWEN21_PRUNA_PADDING',)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    DESCRIPTION = '校验解码尺寸与同源编码记录一致，保留RGB/RGBA；不再缩放，不裁掉生成alpha。'

    def run(self, image, padding):
        from ..ops.qwen21_pruna_encoding import unpad
        return (unpad(image, padding),)
