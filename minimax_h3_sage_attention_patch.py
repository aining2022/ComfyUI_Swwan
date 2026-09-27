# SPDX-License-Identifier: GPL-3.0-only
"""Register the interface without importing the optional MiniMax / GPU backend."""
class MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Swwan/minimax"
    DESCRIPTION = (
        "EXPERIMENTAL: use the KJ-style SageAttention kernel for MiniMax H3 self-attention "
        "to reduce peak VRAM. Requires compatible MiniMax H3, SageAttention, Triton, and CUDA support."
    )
    EXPERIMENTAL = True

    def patch(self, model):
        from .ops.minimax_backend import MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative as Backend
        return Backend().patch(model)
