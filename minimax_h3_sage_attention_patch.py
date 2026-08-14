"""MiniMax H3 SageAttention patch adapted from ComfyUI-KJNodes.

Optional MiniMax H3, SageAttention, and Triton dependencies are intentionally
loaded defensively so this module remains importable on older ComfyUI builds.
"""

import importlib
import logging

import torch


try:
    import comfy.model_management as _model_management
except Exception:
    _model_management = None

try:
    from comfy.ldm.minimax.model import MiniMaxH3Model as _MiniMaxH3Model
except Exception:
    _MiniMaxH3Model = None

try:
    from comfy.quant_ops import ck as _ck
except Exception:
    _ck = None

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


_cuda_archs = None
_sageattention_error = None
per_warp_int8_cuda = None
per_block_int8_triton = None
per_channel_fp8 = None
attn_false = None

try:
    from sageattention.core import (
        attn_false,
        get_cuda_arch_versions,
        per_block_int8_triton,
        per_channel_fp8,
        per_warp_int8_cuda,
    )

    _cuda_archs = get_cuda_arch_versions()
except Exception as error:
    _sageattention_error = error


def _cuda_version():
    try:
        version = torch.version.cuda
        if version is None:
            return 0, 0
        major, minor = version.split(".")
        return int(major), int(minor)
    except Exception:
        return 0, 0


_QATTN_PROBE = {
    "sm80": "qk_int8_sv_f16_accum_f32_attn",
    "sm89": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    "sm90": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
}


def _resolve_qattn(arch):
    try:
        core = importlib.import_module("sageattention.core")
    except Exception:
        return None

    candidates = [getattr(core, f"_qattn_{arch}", None), getattr(core, f"{arch}_compile", None)]
    try:
        module = importlib.import_module(f"sageattention.{arch}_compile")
        candidates.extend([module, getattr(module, f"_qattn_{arch}", None)])
    except Exception:
        pass

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, _QATTN_PROBE[arch]):
            return candidate
    return None


_qattn_sm80 = _resolve_qattn("sm80")
_qattn_sm89 = _resolve_qattn("sm89")
_qattn_sm90 = _resolve_qattn("sm90")
sageplus_sm89_available = (
    _qattn_sm89 is not None
    and hasattr(_qattn_sm89, "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf")
    and _cuda_version() >= (12, 8)
)


if HAS_TRITON:
    # Vendored from SageAttention's quant_per_thread.py. int64 row offsets
    # prevent overflow for MiniMax H3's large packed sequences.
    @triton.jit
    def _quant_query_per_thread_int8_i64_kernel(
        Input,
        Output,
        Scale,
        L,
        stride_iz,
        stride_ih,
        stride_in,
        stride_oz,
        stride_oh,
        stride_on,
        stride_sz,
        stride_sh,
        C: tl.constexpr,
        BLK: tl.constexpr,
    ):
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
        offs_k = tl.arange(0, C)

        input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

        x = tl.load(input_ptrs, mask=offs_n[:, None] < L).to(tl.float32)
        scale = tl.max(tl.abs(x)) / 127.0 + 0.0000001
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        tl.store(output_ptrs, x_int8.to(tl.int8), mask=offs_n[:, None] < L)
        tl.store(scale_ptrs, scale)


    @triton.jit
    def _quant_key_per_thread_int8_i64_kernel(
        Input,
        Output,
        Scale,
        L,
        stride_iz,
        stride_ih,
        stride_in,
        stride_oz,
        stride_oh,
        stride_on,
        stride_sz,
        stride_sh,
        C: tl.constexpr,
        BLK: tl.constexpr,
    ):
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)

        input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L).to(tl.float32)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L).to(tl.float32)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127.0 + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        tl.store(output_ptrs0, x0_int8.to(tl.int8), mask=offs_n0[:, None] < L)
        tl.store(output_ptrs1, x1_int8.to(tl.int8), mask=offs_n1[:, None] < L)
        tl.store(scale_ptrs, scale)


def _per_thread_int8_i64(
    q,
    k,
    km=None,
    BLKQ=128,
    WARPQ=32,
    BLKK=64,
    WARPK=64,
    tensor_layout="NHD",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        batch, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        batch, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty(
        (batch, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
        device=q.device,
        dtype=torch.float32,
    )
    k_scale = torch.empty(
        (batch, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4),
        device=q.device,
        dtype=torch.float32,
    )

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, batch)
    _quant_query_per_thread_int8_i64_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        C=head_dim,
        BLK=WARPQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, batch)
    _quant_key_per_thread_int8_i64_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        C=head_dim,
        BLK=WARPK,
    )

    return q_int8, q_scale, k_int8, k_scale


def _sageattn_int8_fp8_nhd(qkv, dtype):
    # qkv elements use NHD layout: [batch, sequence, heads, head_dim].
    q, k, v = qkv
    qkv.clear()
    head_dim = q.shape[-1]
    tensor_layout = "NHD"
    tensor_layout_code = 0
    is_causal = 0
    qk_quant_granularity = 3
    return_lse = 0
    sm_scale = head_dim**-0.5
    quant_v_scale_max = 448.0
    arch = _cuda_archs[0]

    if arch in {"sm80", "sm86"}:
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q,
            k,
            tensor_layout=tensor_layout,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
        )
        del q, k
        output = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        v_fp16 = v.to(torch.float16)
        del v
        _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(
            q_int8,
            k_int8,
            v_fp16,
            output,
            q_scale,
            k_scale,
            tensor_layout_code,
            is_causal,
            qk_quant_granularity,
            sm_scale,
            return_lse,
        )
    elif arch == "sm75":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(
            q,
            k,
            sm_scale=sm_scale,
            tensor_layout=tensor_layout,
        )
        del q, k
        output, _ = attn_false(
            q_int8,
            k_int8,
            v,
            q_scale,
            k_scale,
            tensor_layout=tensor_layout,
            output_dtype=dtype,
            attn_mask=None,
            return_lse=False,
        )
        del v
    elif arch == "sm89":
        if sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        else:
            pv_accum_dtype = "fp32+fp32"
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q,
            k,
            tensor_layout=tensor_layout,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
        )
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(
            v,
            tensor_layout=tensor_layout,
            scale_max=quant_v_scale_max,
            smooth_v=False,
        )
        del v
        output = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8,
                k_int8,
                v_fp8,
                output,
                q_scale,
                k_scale,
                v_scale,
                tensor_layout_code,
                is_causal,
                qk_quant_granularity,
                sm_scale,
                return_lse,
            )
        else:
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                q_int8,
                k_int8,
                v_fp8,
                output,
                q_scale,
                k_scale,
                v_scale,
                tensor_layout_code,
                is_causal,
                qk_quant_granularity,
                sm_scale,
                return_lse,
            )
        del v_fp8, v_scale
    elif arch == "sm90":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(
            q,
            k,
            tensor_layout=tensor_layout,
            BLKQ=64,
            WARPQ=16,
            BLKK=128,
            WARPK=128,
        )
        del q, k
        # SageAttention's sm90 kernel requires CTA_K=128. Its FP8 helper only
        # pads to 64, so long MiniMax H3 prompts need this explicit V padding.
        kv_len = v.size(1)
        v_pad_len = 128 - (kv_len % 128) if kv_len % 128 else 0
        if v_pad_len:
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        v.size(0),
                        v_pad_len,
                        v.size(2),
                        v.size(3),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=1,
            )
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
        del v
        output = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            output,
            q_scale,
            k_scale,
            v_scale,
            tensor_layout_code,
            is_causal,
            qk_quant_granularity,
            sm_scale,
            return_lse,
        )
        del v_fp8, v_scale
    elif arch in {"sm120", "sm121"}:
        if sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        else:
            pv_accum_dtype = "fp32"
        qk_quant_granularity = 2
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q,
            k,
            km=k.mean(dim=1, keepdim=True),
            tensor_layout=tensor_layout,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
        )
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(
            v,
            tensor_layout=tensor_layout,
            scale_max=quant_v_scale_max,
            smooth_v=False,
        )
        del v
        output = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
                q_int8,
                k_int8,
                v_fp8,
                output,
                q_scale,
                k_scale,
                v_scale,
                tensor_layout_code,
                is_causal,
                qk_quant_granularity,
                sm_scale,
                return_lse,
            )
        else:
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                q_int8,
                k_int8,
                v_fp8,
                output,
                q_scale,
                k_scale,
                v_scale,
                tensor_layout_code,
                is_causal,
                qk_quant_granularity,
                sm_scale,
                return_lse,
            )
        del v_fp8, v_scale
    else:
        raise RuntimeError(f"Unsupported SageAttention CUDA architecture: {arch}")

    del q_int8, q_scale, k_int8, k_scale
    return output


def minimax_sageattn_forward(self, x, rope_freqs=None, transformer_options={}):
    # MiniMax H3 uses an unbatched packed sequence. The list form is produced
    # by MiniMaxLowVRAMAttention and deliberately transfers ownership of x.
    if isinstance(x, list):
        x = x.pop()
    dtype = x.dtype
    device = x.device
    sequence_length = x.shape[0]

    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    del x
    q = q.view(1, sequence_length, self.heads, self.head_dim)
    k = k.view(1, sequence_length, self.heads, self.head_dim)
    v = v.view(1, sequence_length, self.heads, self.head_dim)

    if rope_freqs is not None:
        q_weight = _model_management.cast_to(self.q_norm.weight, device=device)
        k_weight = _model_management.cast_to(self.k_norm.weight, device=device)
        _ck.rms_rope_split_half_(
            q,
            k,
            rope_freqs,
            q_weight,
            k_weight,
            epsilon=self.q_norm.eps,
            rot_dim=rope_freqs.shape[-3] * 2,
        )
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)

    head_chunks = (
        min(transformer_options.get("minimax_head_chunks", 1), self.heads)
        if isinstance(transformer_options, dict)
        else 1
    )
    if head_chunks <= 1:
        qkv = [q, k, v]
        del q, k, v
        output = _sageattn_int8_fp8_nhd(qkv, dtype)
        return self.out_proj(output.view(sequence_length, self.heads * self.head_dim))

    output = torch.empty(
        (sequence_length, self.heads * self.head_dim),
        dtype=dtype,
        device=device,
    )
    output_nhd = output.view(1, sequence_length, self.heads, self.head_dim)
    head_start = 0
    for chunk_index in range(head_chunks):
        head_end = head_start + self.heads // head_chunks + (1 if chunk_index < self.heads % head_chunks else 0)
        output_nhd[:, :, head_start:head_end] = _sageattn_int8_fp8_nhd(
            [q[:, :, head_start:head_end], k[:, :, head_start:head_end], v[:, :, head_start:head_end]],
            dtype,
        )
        head_start = head_end
    del q, k, v
    return self.out_proj(output)


def _require_runtime():
    if _sageattention_error is not None:
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires a compatible "
            "SageAttention installation with its core quantization kernels."
        ) from _sageattention_error
    if not HAS_TRITON:
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires the optional "
            "Triton package for its int64-safe Q/K quantization kernels."
        )
    if _MiniMaxH3Model is None:
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires a ComfyUI build "
            "with MiniMax H3 support (comfy.ldm.minimax.model.MiniMaxH3Model)."
        )
    if _ck is None or not hasattr(_ck, "rms_rope_split_half_"):
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires ComfyUI's "
            "comfy.quant_ops.ck.rms_rope_split_half_ kernel."
        )
    if _model_management is None or not hasattr(_model_management, "cast_to"):
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires "
            "comfy.model_management.cast_to from a compatible ComfyUI build."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) requires a CUDA-enabled "
            "PyTorch runtime."
        )
    if not _cuda_archs:
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) could not determine the "
            "SageAttention CUDA architecture."
        )

    arch = _cuda_archs[0]
    if arch not in {"sm75", "sm80", "sm86", "sm89", "sm90", "sm120", "sm121"}:
        raise RuntimeError(
            "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) does not support "
            f"SageAttention CUDA architecture {arch}."
        )
    if arch in {"sm80", "sm86"} and _qattn_sm80 is None:
        raise RuntimeError("SageAttention's sm80 fused attention kernel is unavailable.")
    if arch == "sm89" and _qattn_sm89 is None:
        raise RuntimeError("SageAttention's sm89 fused attention kernel is unavailable.")
    if arch == "sm90" and _qattn_sm90 is None:
        raise RuntimeError("SageAttention's sm90 fused attention kernel is unavailable.")
    if arch in {"sm120", "sm121"} and _qattn_sm89 is None:
        raise RuntimeError("SageAttention's sm89-compatible fused attention kernel is unavailable.")


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
        _require_runtime()
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, _MiniMaxH3Model):
            raise RuntimeError(
                "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) can only be applied "
                "to a MiniMax H3 model."
            )

        logging.info("Applying MiniMax H3 Memory Efficient Sage Attention Patch (KJ Alternative) to all transformer blocks")
        for index, block in enumerate(diffusion_model.blocks):
            model_clone.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                minimax_sageattn_forward.__get__(block.attn, block.attn.__class__),
            )

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative": MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative": "MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
