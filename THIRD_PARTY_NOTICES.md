# Code, assets and licenses

The combined distribution is GPL-3.0-only (root `LICENSE`, `licenses/GPL-3.0.txt`). This does not relicense MIT source files or remove their original copyright notices. All source and fixture copies are included as source, with local modifications described below. Machine-readable per-node origin/license fields are maintained in `node_manifest.json`.

## Sources

| Source / original copyright | Included implementation | Evidence / revision | Original license |
| --- | --- | --- | --- |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), kijai and contributors | `nodes/{resize,mask,batch,concat,color,io,device,model,transition}.py`, `ops/{image_common,transitions,grid,minimax_backend}.py`, attention patches, BlockifyMask | Existing imported code retained; local reference `6ab7e8130e449ed2c0037589bcf84146ceb7fc9c`. Historical import revision for all older code was not recorded. | GPL-3.0, `licenses/GPL-3.0.txt` |
| [Goohaitools-comfyui](https://github.com/goohai/Goohaitools-comfyui), goohai and contributors | `edit_region.py`, `edit_image_ops.py`, range resize and color conversion in `workflow_tools.py` | `a84303e6e73a289af59d96eddab9f521ebd00643` | GPL-3.0, `licenses/GPL-3.0.txt` |
| [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle), Copyright (c) 2024 chflame163 | crop/restore V1–V5 modules, `color_image.py`, `image_blend.py`, `image_scale_by_aspect_ratio_v2.py`, `layerstyle_utils.py`, `ops/aspect_resize.py` | Existing imports retained; local license/reference `a3459a7638c4c2839878089c105c73af0eb2edd2`. Exact historical import revision not recorded. | MIT, `licenses/MIT-LayerStyle.txt` |
| [ComfyUI-Apt_Preset](https://github.com/cardenluo/ComfyUI-Apt_Preset), original notice Copyright (c) 2023 pythongosssss | `nodes/{math_utils,scheduling,data_lists}.py`, `ops/workflow_helpers.py`, `C_math.py`, `image_resize_sum.py`, `main_unit.py` compatibility exports, `ops/{types,easing,conversion,math_presets}.py` | Existing imports retained; local license reference `a61317df3ecdfd3d899a1e9a1a7cebd5bfa765c1`. Exact historical import revision not recorded; the local checkout is a fork and its README identifies cardenluo upstream. | MIT, `licenses/MIT-Apt.txt` |
| [ComfyUI-UniversalToolkit](https://github.com/whmc76/ComfyUI-UniversalToolkit), Copyright (c) 2024 UniversalToolkit | `math_expression.py` interface and named-widget/size lookup | Repository import commit `890655a`; upstream historical revision not recorded. Current official MIT notice retained. | MIT, `licenses/MIT-UniversalToolkit.txt` |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy), Copyright (c) 2023 Regis Gaughan, III (rgthree) | `seed.py` backend interface and special-seed fallback | Existing import retained; coexistence/license reference `449c58fcdd612f7733e54c51f6758ead63fa180b`. Original frontend dependency chain removed. | MIT, `licenses/MIT-rgthree.txt` |
| [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui), Jordan Thompson | `bounded_image_crop.py`; RGB conversion behavior reference; `tests/fixtures/was_tensors.py` | Local reference `eb772edaa8c4eeca459839435d96b750e40e927f`; historical bounded import revision not recorded. | MIT, `licenses/MIT-WAS.txt` |
| Swwan, Copyright (c) 2025 aining2022 | independent batch/RGBA/save/switch/color-fix utilities, registry/migration/catalog/scripts/skill and frontend | Original repository notice preserved before distribution license update. New saver/grid link GPL source where indicated. | MIT for original independent files, `licenses/MIT-Swwan.txt`; SPDX headers for GPL additions |

Scheduling helpers also have a Comfyroll Studio ancestry indicated by retained CR messages and matching helper names ([RockOfFire and Akatsuzi/Suzie1 source](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/blob/main/nodes/functions_animation.py)). Their modified form was received in the MIT-licensed Apt source; the original historical import/permission chain was not recorded. This audit retains the received MIT notice and the ancestor credit rather than inventing an upstream license or commit. No new Comfyroll source was imported in this remediation.

No source files import third-party plugin packages at runtime. Imports of ComfyUI itself are host interfaces. The independent COLORCODE frontend does not copy the AILab widget. New seed/dynamic/preview frontend code is independent Swwan code.

## Modifications in 1.0.0

Node IDs/categories are centrally declared. KJ image code was split mechanically by task; historical Python modules forward named exports. CPU branches retain original algorithms. Optional image/GPU dependencies are deferred to execution. Empty-input/broadcast bugs are fixed; default crop/resize/concat branches remain. Crop/restore, resize, blend, math and grid add explicit modes or outputs. Old aspect and fixed-grid wrappers share algorithms. Saving shares encoding, metadata and path helpers while historical interfaces retain their naming rules. Math Calculate no longer evaluates arbitrary Python; its supported mathematical presets and fallback contract remain.

The Qwen algorithms preserve SEAM crop/restore boundaries, padding and interpolation; restore returns the actual full-canvas composition mask. RGB conversion preserves reference PIL quantization/dynamic-range folding and extends gray/RGB/RGBA batches. Range resize handles a mask-only black input. These changes are tested with frozen local reference classes.

## Fixtures and assets

- `tests/fixtures/reference/` contains frozen Goohaitools/KJ class implementations under GPL-3.0. `was_tensors.py` and `was_dynamic.py` are MIT WAS code; the fixture import path alone is redirected locally. `baseline_*.py` preserves original Swwan working interfaces/branches (source checkpoint `7cfbeba`); KJ baseline is GPL, LayerStyle/Apt baselines MIT. Their purpose is portable CPU regression, not importing installed plugins or reading Downloads.
- `fonts/FreeMono.ttf` and `FreeMonoBoldOblique.otf` are GNU FreeFont, Copyleft 2002, 2003, 2005, 2008, 2009, 2010 Free Software Foundation. Embedded notices use GPL-3.0-or-later with the font embedding exception; see `licenses/FreeFont.txt`. Font notices were extracted from the distributed font metadata.
- `TTNorms-Black.otf` is removed from this distribution because a redistributable license was not established. Historical `TTNorms-Black.otf` values are accepted and resolve to FreeMono; migration writes `FreeMono.ttf`. Typography changes intentionally; workflow loading remains supported.
- Qwen2511 workflow is the user-supplied workflow with image-processing node migration. Its model files are not included; model names do not grant model licenses. Other CPU examples are authored for this project and contain no models or third-party media.
- `.DS_Store` is pre-existing local metadata, preserved in this checkout but excluded from distribution packaging.

Historical import revisions that were not recorded are explicitly identified above rather than assigned the current reference SHA. Keep these notices and original license files with redistributed source.
