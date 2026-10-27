# Qwen Image 2.1 Turbo integration

`SwwanQwen21Controls` → `SwwanQwen21References` → `SwwanQwen21Dispatch` provides one task selector and one absolute LoRA strength (0–2, default 1). Shared UNET/CLIP/VAE loaders feed the dispatcher. The active task expands into preprocessing, one shared inference implementation and restoration; the face task expands to one person at a time with stable original-image region mapping. Rows are ordered top-to-bottom, and people left-to-right within each row; the preview and reference lookup share the frozen SEGS.

`SwwanCropByMaskV5` defaults to its historical `bounds` mode. New `qwen21_aligned` mode uses resolution 1024 / tolerance 1 / 32-pixel alignment, appends three outputs and leaves all previous output slots intact. The paired Restore V4 recognizes the versioned SEAM. The encoder consumes already prepared images without rescaling; a latent is created from the actual first encoded reference's shape.

Generic task IDs: remove, watermark, edit, local_edit, enhance, cutout, face_swap, body_swap, outfit_swap, outpaint, angles. Edit/local_edit require a nonblank extra prompt. Consistency auto means local on / full-image off. A zero LoRA strength bypasses patch loading. Switching tasks never changes the public strength. All persons share extra prompt and strength.

Face processing requires the existing v5 dependencies (Easy-Use segmentation, Goohai alignment, RMBG and the utility nodes in the frozen graph). Generic branches do not import or execute those detectors. `QwenImage21SageAttentionT8` is required for GPU inference; its optional BlockCache and Spectrum are not applied.

Per-run diagnostic artifacts in `output/qwen21-unified/<run_id>/` contain request, exact prompt, sigma array, references, raw result, final result, masks and timing. Failed face tasks stop before subsequent people and retain prior completed canvases. A successful queue entry alone is not quality acceptance.

Integration deliverables and L20 measurements are maintained in the user's `replicate-models/image/qwen21-unified` project. No online endpoint is replaced by installing these nodes. Model downloads and quality acceptance are separate from registration and CPU contracts.
