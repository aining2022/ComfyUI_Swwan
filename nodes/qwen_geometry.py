# SPDX-License-Identifier: MIT
"""Expose only the recorded geometry inverse, without restoration or inference."""
from ..ops.qwen21_geometry import inverse_aligned


class QwenGeometryUnpack:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'seam': ('SEAM',)},
                'optional': {'require_alpha': ('BOOLEAN', {'default': False,
                    'tooltip': '抠图时开启；若生成图只有RGB则明确报错，不伪造透明背景。'})}}

    RETURN_TYPES = ('IMAGE', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('patch', 'x', 'y', 'width', 'height', 'original_width', 'original_height')
    FUNCTION = 'unpack'
    DESCRIPTION = ('仅反变换：使用同一 Swwan Qwen aligned SEAM 去补边并还原局部尺寸，保留所有通道。'
                   '输出局部图和原图坐标；不执行模型、校色、遮罩融合或贴回。不要连接 GH SEAM。')

    def unpack(self, image, seam, require_alpha=False):
        if not isinstance(seam, dict) or seam.get('qwen21_geometry_version') != 1:
            raise ValueError('Expected the same Swwan Qwen aligned SEAM, not a GH seam')
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError('Expected a BHWC RGB or RGBA image')
        if require_alpha and image.shape[-1] != 4:
            raise ValueError('Cutout requires native RGBA output from the Qwen VAE')
        patch = inverse_aligned(image, seam)
        x, y, right, bottom = seam['box']
        width, height = seam['original_size']
        return patch, x, y, right-x, bottom-y, width, height
