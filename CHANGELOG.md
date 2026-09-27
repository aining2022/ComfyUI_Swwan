# Changelog

## Unreleased

- Split documentation into a task-oriented user README and an AGENTS.md development entry with catalog, compatibility, migration and validation guidance. Include both entries in distributions and refresh the installation checklist.

- Integrate 43 pure image/mask instances from the Qwen face/head workflow while preserving all model, prompt, sampler, loop and explicitly excluded LG nodes.
- Add five focused Advanced tools: Mask Process, Mask Combine, Mask Analyze, Mask Segments and Image Matte. SEGS remains a seven-field Impact-compatible contract, with no detector/model imports.
- Extend existing crop with direct linked edit-region factors, Resize with the original Essentials algorithm, Color Match with six-space mean/std matching, and Preview with numbered mask regions. Existing modes, defaults and output slots stay intact.
- Retain WAS single-mask `[1,1,H,W]` filling output and repair its ambiguous multi-mask path with per-image processing. Binary mode normalizes MASK channels exactly as Impact does.
- Add an idempotent, source-preserving workflow migration and frozen CPU references. Replace the unavailable preview font with redistributed FreeMono; enabled numbering intentionally changes typography.

## 1.0.0 — 2026-09-26

- 113 registered capabilities organized into 21 primary, 72 Advanced, 18 Legacy and 2 Experimental entries. Unique manifest and duplicate-ID rejection; all displays carry `(Swwan)`.
- Breaking namespace migration: 56 KJ overlaps gain `Swwan` + old ID, 4 LayerStyle overlaps use independent crop/aspect/restore IDs, seed becomes `SwwanSeed`. Other IDs, including `ImageResizeByMegapixels`, remain. No conflict aliases. UI/API migration requires explicit ownership and preserves the original file.
- Qwen image processing integrated into crop/restore, resize and blend modes; core nodes, QwenEditUtils and generation parameters unchanged. Full SEAM and native MASK appended to crop; BOOLEAN appended after Math INT/FLOAT.
- Corrected Resize v2 keep_proportion default from invalid Boolean false to enum value stretch. Generic migration repairs the same invalid historical literal; valid options and algorithms remain.
- Fixed single-mask broadcasting, normal crop empty-mask passthrough and unequal-batch-size error, two empty outputs for batch mask crop, and a clear empty-list error.
- Unified multi-format saver, alpha and workflow metadata, captions/JSON and safe per-folder numbering. Legacy interfaces preserved; `SaveImageKJ` relative folders now resolve under ComfyUI output (old unintended cwd path repaired). New saver relative paths always use output; old IO wrappers keep their established cwd-relative path behavior.
- Multi-input concat gains grid and batch-grid; aspect presets share the original ratio algorithm; AST math presets replace arbitrary Python evaluation while preserving Calculate output order/error zeros. Unsupported Python expressions are an intentional security/behavior restriction.
- Independent frontend for seed controls, dynamic inputs and FastPreview; hidden mode controls retain saved values and connections. Seed prompt/output workflow metadata records the actual seed; API special seeds retain random fallback.
- Split image algorithms by task; explicit compatibility exports replace wildcard imports. Optional vision/GPU imports are deferred to relevant execution.
- GPL-3.0 combined distribution with original source notices; TTNorms removed and historical values map to FreeMono. CPU fixtures, fixed ComfyUI CI reference, contribution guide and installed node-development skill.

No model inference, GPU validation or public release is implied by local CPU acceptance. Hardware checks are listed separately.
