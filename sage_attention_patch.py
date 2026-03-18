import importlib.util
import logging

import torch
from comfy.ldm.modules.attention import wrap_attn


SAGE_ATTN_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
    "sageattn3",
    "sageattn3_per_block_mean",
]


def _module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def _disable_compile_if_needed(func, allow_compile):
    if allow_compile:
        return func

    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None) if compiler is not None else None
    if disable is None:
        return func

    return disable()(func)


def _cuda_arch_for_tensor(tensor):
    if not tensor.is_cuda:
        return None

    major, minor = torch.cuda.get_device_capability(tensor.device)
    return f"sm{major}{minor}"


def _raise_missing_backend(selected_mode):
    if "sageattn3" in selected_mode:
        raise RuntimeError(
            f"{selected_mode} requires the optional 'sageattn3' package, "
            "but it is not installed in the current ComfyUI environment."
        )

    raise RuntimeError(
        f"{selected_mode} requires the optional 'sageattention' package, "
        "but it is not installed in the current ComfyUI environment."
    )


def get_sage_attention_func(selected_mode, allow_compile=False):
    logging.info("Using sage attention mode: %s", selected_mode)

    if "sageattn3" in selected_mode:
        if not _module_available("sageattn3"):
            _raise_missing_backend(selected_mode)
    elif not _module_available("sageattention"):
        _raise_missing_backend(selected_mode)

    warned = set()

    def warn_once(key, message):
        if key in warned:
            return
        logging.warning(message)
        warned.add(key)

    def mask_fallback(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        warn_once(
            ("mask", selected_mode),
            f"{selected_mode} does not support attn_mask in the current SageAttention backend; "
            "falling back to sageattn_qk_int8_pv_fp16_triton.",
        )
        return sageattn_qk_int8_pv_fp16_triton(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            tensor_layout=tensor_layout,
        )

    def fp8_cuda_dispatch(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=None,
        tensor_layout="NHD",
        pv_accum_dtype="fp32+fp32",
    ):
        arch = _cuda_arch_for_tensor(q)
        if arch == "sm90":
            from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

            if pv_accum_dtype != "fp32+fp32":
                warn_once(
                    ("sm90-accum", selected_mode),
                    "H100 sm90 FP8 kernel only supports pv_accum_dtype='fp32+fp32'; "
                    "overriding the requested accumulation mode.",
                )
            if selected_mode != "sageattn_qk_int8_pv_fp8_cuda_sm90":
                warn_once(
                    ("sm90-remap", selected_mode),
                    f"{selected_mode} remapped to sageattn_qk_int8_pv_fp8_cuda_sm90 on Hopper.",
                )
            return sageattn_qk_int8_pv_fp8_cuda_sm90(
                q,
                k,
                v,
                is_causal=is_causal,
                tensor_layout=tensor_layout,
                pv_accum_dtype="fp32+fp32",
            )

        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        return sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            pv_accum_dtype=pv_accum_dtype,
            tensor_layout=tensor_layout,
        )

    supports_mask = False
    if selected_mode == "auto":
        from sageattention import sageattn

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif selected_mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32",
                tensor_layout=tensor_layout,
            )
    elif selected_mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                tensor_layout=tensor_layout,
            )

        supports_mask = True
    elif selected_mode == "sageattn_qk_int8_pv_fp8_cuda":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return fp8_cuda_dispatch(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32+fp32",
                tensor_layout=tensor_layout,
            )
    elif selected_mode == "sageattn_qk_int8_pv_fp8_cuda++":
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return fp8_cuda_dispatch(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32+fp16",
                tensor_layout=tensor_layout,
            )
    elif selected_mode == "sageattn_qk_int8_pv_fp8_cuda_sm90":
        from sageattention import sageattn

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            arch = _cuda_arch_for_tensor(q)
            if arch != "sm90":
                warn_once(
                    ("sm90-arch", selected_mode),
                    f"{selected_mode} requested on {arch or 'non-cuda'}; falling back to auto kernel selection.",
                )
                return sageattn(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)
            return fp8_cuda_dispatch(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)
    elif "sageattn3" in selected_mode:
        from sageattn3 import sageattn3_blackwell

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                per_block_mean=(selected_mode == "sageattn3_per_block_mean"),
            )
            return out.transpose(1, 2) if tensor_layout == "NHD" else out

        supports_mask = True
    else:
        raise ValueError(f"Unsupported sage attention mode: {selected_mode}")

    sage_func = _disable_compile_if_needed(sage_func, allow_compile)
    mask_fallback = _disable_compile_if_needed(mask_fallback, allow_compile)

    @wrap_attn
    def attention_sage(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        if skip_reshape:
            batch, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            batch, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(lambda tensor: tensor.view(batch, -1, heads, dim_head), (q, k, v))
            tensor_layout = "NHD"

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        selected_func = sage_func if (mask is None or supports_mask) else mask_fallback
        out = selected_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(input_dtype)

        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(batch, -1, heads * dim_head)

        return out

    return attention_sage


class PatchSageAttentionKJ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sage_attention": (
                    SAGE_ATTN_MODES,
                    {
                        "default": "disabled",
                        "tooltip": "Patch the model attention override to use SageAttention. "
                                   "Choose disabled to remove an existing SageAttention override.",
                    },
                ),
            },
            "optional": {
                "allow_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow torch.compile inside the SageAttention wrapper when the backend supports it.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Swwan/model"
    DESCRIPTION = (
        "Migrate Patch Sage Attention KJ into ComfyUI_Swwan. "
        "Requires the optional sageattention or sageattn3 package in the ComfyUI runtime."
    )
    EXPERIMENTAL = True

    def patch(self, model, sage_attention, allow_compile=False):
        model_clone = model.clone()
        transformer_options = model_clone.model_options.setdefault("transformer_options", {})

        if sage_attention == "disabled":
            transformer_options.pop("optimized_attention_override", None)
            return (model_clone,)

        new_attention = get_sage_attention_func(sage_attention, allow_compile=allow_compile)

        def attention_override_sage(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        transformer_options["optimized_attention_override"] = attention_override_sage
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "PatchSageAttentionKJ": PatchSageAttentionKJ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchSageAttentionKJ": "Patch Sage Attention KJ (Swwan)",
}
