# 节点全量清单

由唯一注册清单和实际接口生成；层级已应用到菜单。

当前注册 **118** 个节点，**15** 个分类。

| 层级 | 数量 |
| --- | --- |
| 兼容入口 | 18 |
| 主入口 | 21 |
| 专用工具 | 77 |
| 实验功能 | 2 |

| 层级 | 节点 ID | 显示名 | 当前分类 | 输出 | 源码 |
| --- | --- | --- | --- | --- | --- |
| 兼容入口 | `SwwanCropByMaskV2` | CropByMask V2 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, MASK, BOX, IMAGE | [crop_by_mask_v2.py](../crop_by_mask_v2.py#L10) |
| 兼容入口 | `SwwanCropByMaskV3` | CropByMask V3 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, IMAGE, BOX, IMAGE | [crop_by_mask_v3.py](../crop_by_mask_v3.py#L10) |
| 兼容入口 | `LayerUtility: CropByMask V4` | CropByMask V4 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, IMAGE, BOX, IMAGE | [crop_by_mask_v4.py](../crop_by_mask_v4.py#L14) |
| 主入口 | `SwwanCropByMaskV5` | Mask Crop (Swwan) | Swwan/Image | IMAGE, IMAGE, BOX, IMAGE, SEAM, MASK | [crop_by_mask_v5.py](../crop_by_mask_v5.py#L14) |
| 兼容入口 | `SwwanRestoreCropBox` | RestoreCropBox (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, MASK | [restore_crop_box.py](../restore_crop_box.py#L9) |
| 兼容入口 | `LayerUtility: RestoreCropBox V2` | RestoreCropBox V2 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, MASK | [restore_crop_box_v2.py](../restore_crop_box_v2.py#L11) |
| 兼容入口 | `SwwanRestoreCropBoxV3` | Restore Crop Box V3 (Batch) (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, MASK | [restore_crop_box_v3.py](../restore_crop_box_v3.py#L11) |
| 主入口 | `SwwanRestoreCropBoxV4` | Restore Crop (Swwan) | Swwan/Image | IMAGE, MASK | [restore_crop_box_v4.py](../restore_crop_box_v4.py#L11) |
| 兼容入口 | `SwwanImageScaleByAspectRatioV2` | ImageScaleByAspectRatio V2 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, MASK, BOX, INT, INT | [image_scale_by_aspect_ratio_v2.py](../image_scale_by_aspect_ratio_v2.py#L5) |
| 主入口 | `LayerUtility: ColorImage (Swwan)` | ColorImage (Swwan) | Swwan/Image | IMAGE | [color_image.py](../color_image.py#L7) |
| 主入口 | `ImageBlendSwwan` | Image Blend (Swwan) | Swwan/Image | IMAGE | [image_blend.py](../image_blend.py#L13) |
| 主入口 | `SwwanSeed` | Seed (Swwan) | Swwan/Utils | INT | [seed.py](../seed.py#L17) |
| 主入口 | `SwwanImageListToImageBatch` | Image List to Image Batch (Swwan) | Swwan/Batch | IMAGE | [image_batch_utils.py](../image_batch_utils.py#L7) |
| 主入口 | `SwwanImageBatchToImageList` | Image Batch to Image List (Swwan) | Swwan/Batch | IMAGE | [image_batch_utils.py](../image_batch_utils.py#L91) |
| 专用工具 | `Mask_transform_sum` | Mask Transform Sum (Swwan) | Swwan/Advanced/Mask | IMAGE, MASK | [image_resize_sum.py](../image_resize_sum.py#L30) |
| 专用工具 | `Image_Resize_sum` | Image Resize Sum (Swwan) | Swwan/Advanced/Image | IMAGE, MASK, STITCH3, FLOAT | [image_resize_sum.py](../image_resize_sum.py#L349) |
| 专用工具 | `Image_Resize_sum_restore` | Image Resize Sum Restore (Swwan) | Swwan/Advanced/Image | IMAGE, MASK, IMAGE | [image_resize_sum.py](../image_resize_sum.py#L738) |
| 专用工具 | `Image_Resize_sum_data` | Image Resize Sum Data (Swwan) | Swwan/Advanced/Image | INT, INT, INT, INT, INT, INT, INT, INT, INT, INT, FLOAT | [image_resize_sum.py](../image_resize_sum.py#L269) |
| 专用工具 | `math_Remap_data` | Math Remap Data (Swwan) | Swwan/Advanced/Utils | FLOAT, INT | [nodes/math_utils.py](../nodes/math_utils.py#L14) |
| 兼容入口 | `math_calculate` | Math Calculate (Swwan) · 旧版兼容 | Swwan/Legacy | FLOAT, INT, BOOLEAN | [nodes/math_utils.py](../nodes/math_utils.py#L69) |
| 专用工具 | `list_Slice` | List Slice (Swwan) | Swwan/Advanced/Batch | * | [nodes/data_lists.py](../nodes/data_lists.py#L14) |
| 专用工具 | `list_Merge` | List Merge (Swwan) | Swwan/Advanced/Batch | * | [nodes/data_lists.py](../nodes/data_lists.py#L58) |
| 专用工具 | `list_Value` | List Value (Swwan) | Swwan/Advanced/Batch | FLOAT, INT, FLOAT | [nodes/data_lists.py](../nodes/data_lists.py#L93) |
| 专用工具 | `list_num_range` | List Num Range (Swwan) | Swwan/Advanced/Batch | FLOAT, LIST, INT | [nodes/data_lists.py](../nodes/data_lists.py#L155) |
| 专用工具 | `sch_split_text` | Schedule Split Text (Swwan) | Swwan/Advanced/Scheduling | STRING, INT | [nodes/scheduling.py](../nodes/scheduling.py#L15) |
| 专用工具 | `sch_text` | Schedule Text (Swwan) | Swwan/Advanced/Scheduling | STRING, STRING, FLOAT | [nodes/scheduling.py](../nodes/scheduling.py#L86) |
| 专用工具 | `sch_Value` | Schedule Value (Swwan) | Swwan/Advanced/Scheduling | INT, FLOAT, FLOAT, INT | [nodes/scheduling.py](../nodes/scheduling.py#L136) |
| 专用工具 | `sch_Prompt` | Schedule Prompt (Swwan) | Swwan/Advanced/Scheduling | CONDITIONING | [nodes/scheduling.py](../nodes/scheduling.py#L214) |
| 专用工具 | `sch_image` | Schedule Image (Swwan) | Swwan/Advanced/Scheduling | IMAGE | [nodes/scheduling.py](../nodes/scheduling.py#L272) |
| 专用工具 | `sch_mask` | Schedule Mask (Swwan) | Swwan/Advanced/Scheduling | MASK | [nodes/scheduling.py](../nodes/scheduling.py#L295) |
| 专用工具 | `BatchSlice` | Batch Slice (Swwan) | Swwan/Advanced/Batch | * | [nodes/data_lists.py](../nodes/data_lists.py#L180) |
| 专用工具 | `MergeBatch` | Merge Batch (Swwan) | Swwan/Advanced/Batch | LIST | [nodes/data_lists.py](../nodes/data_lists.py#L235) |
| 专用工具 | `type_AnyIndex` | Type Any Index (Swwan) | Swwan/Advanced/Batch | * | [nodes/data_lists.py](../nodes/data_lists.py#L266) |
| 专用工具 | `SwwanImagePass` | Image Pass (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/batch.py](../nodes/batch.py#L6) |
| 专用工具 | `SwwanColorMatch` | Color Match (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/color.py](../nodes/color.py#L5) |
| 兼容入口 | `SwwanSaveImageWithAlpha` | Save Image With Alpha (Swwan) · 旧版兼容 | Swwan/Legacy | 输出节点 | [nodes/io.py](../nodes/io.py#L9) |
| 兼容入口 | `SwwanImageConcanate` | Image Concatenate (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE | [nodes/concat.py](../nodes/concat.py#L6) |
| 兼容入口 | `SwwanImageConcatFromBatch` | Image Concat From Batch (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE | [nodes/concat.py](../nodes/concat.py#L104) |
| 兼容入口 | `SwwanImageGridComposite2x2` | Image Grid Composite 2x2 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE | [nodes/concat.py](../nodes/concat.py#L246) |
| 兼容入口 | `SwwanImageGridComposite3x3` | Image Grid Composite 3x3 (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE | [nodes/concat.py](../nodes/concat.py#L266) |
| 专用工具 | `SwwanImageBatchTestPattern` | Image Batch Test Pattern (Swwan) | Swwan/Advanced/IO | IMAGE | [nodes/io.py](../nodes/io.py#L191) |
| 实验功能 | `SwwanImageGrabPIL` | Image Grab PIL (Swwan) | Swwan/Experimental | IMAGE | [nodes/device.py](../nodes/device.py#L5) |
| 实验功能 | `SwwanWebcamCaptureCV2` | Webcam Capture CV2 (Swwan) | Swwan/Experimental | IMAGE | [nodes/device.py](../nodes/device.py#L54) |
| 专用工具 | `SwwanAddLabel` | Add Label (Swwan) | Swwan/Advanced/IO | IMAGE | [nodes/io.py](../nodes/io.py#L76) |
| 专用工具 | `SwwanGetImageSizeAndCount` | Get Image Size & Count (Swwan) | Swwan/Advanced/Image | IMAGE, INT, INT, INT | [nodes/batch.py](../nodes/batch.py#L26) |
| 专用工具 | `SwwanGetLatentSizeAndCount` | Get Latent Size & Count (Swwan) | Swwan/Advanced/Batch | LATENT, INT, INT, INT, INT, INT | [nodes/batch.py](../nodes/batch.py#L52) |
| 专用工具 | `SwwanImageBatchRepeatInterleaving` | Image Batch Repeat Interleaving (Swwan) | Swwan/Advanced/Batch | IMAGE, MASK | [nodes/batch.py](../nodes/batch.py#L82) |
| 专用工具 | `SwwanImageUpscaleWithModelBatched` | Image Upscale With Model Batched (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/model.py](../nodes/model.py#L5) |
| 专用工具 | `SwwanImageNormalize_Neg1_To_1` | Image Normalize -1 to 1 (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/color.py](../nodes/color.py#L108) |
| 专用工具 | `SwwanRemapImageRange` | Remap Image Range (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/color.py](../nodes/color.py#L126) |
| 专用工具 | `SwwanSplitImageChannels` | Split Image Channels (Swwan) | Swwan/Advanced/Image | IMAGE, IMAGE, IMAGE, MASK | [nodes/color.py](../nodes/color.py#L152) |
| 专用工具 | `SwwanMergeImageChannels` | Merge Image Channels (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/color.py](../nodes/color.py#L182) |
| 专用工具 | `SwwanImagePadForOutpaintMasked` | Image Pad For Outpaint Masked (Swwan) | Swwan/Advanced/Mask | IMAGE, MASK | [nodes/mask.py](../nodes/mask.py#L5) |
| 专用工具 | `SwwanImagePadForOutpaintTargetSize` | Image Pad For Outpaint Target Size (Swwan) | Swwan/Advanced/Image | IMAGE, MASK | [nodes/mask.py](../nodes/mask.py#L94) |
| 专用工具 | `SwwanImagePrepForICLora` | Image Prep For IC Lora (Swwan) | Swwan/Advanced/Image | IMAGE, MASK | [nodes/mask.py](../nodes/mask.py#L153) |
| 专用工具 | `SwwanImageAndMaskPreview` | Image And Mask Preview (Swwan) | Swwan/Advanced/IO | IMAGE | [nodes/io.py](../nodes/io.py#L241) |
| 专用工具 | `SwwanCrossFadeImages` | Cross Fade Images (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/transition.py](../nodes/transition.py#L6) |
| 专用工具 | `SwwanCrossFadeImagesMulti` | Cross Fade Images Multi (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/transition.py](../nodes/transition.py#L62) |
| 专用工具 | `SwwanTransitionImagesMulti` | Transition Images Multi (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/transition.py](../nodes/transition.py#L116) |
| 专用工具 | `SwwanTransitionImagesInBatch` | Transition Images In Batch (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/transition.py](../nodes/transition.py#L185) |
| 专用工具 | `SwwanImageBatchJoinWithTransition` | Image Batch Join With Transition (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/transition.py](../nodes/transition.py#L244) |
| 专用工具 | `SwwanShuffleImageBatch` | Shuffle Image Batch (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L121) |
| 主入口 | `SwwanGetImageRangeFromBatch` | Get Image Range From Batch (Swwan) | Swwan/Batch | IMAGE, MASK | [nodes/batch.py](../nodes/batch.py#L143) |
| 专用工具 | `SwwanImageBatchExtendWithOverlap` | Image Batch Extend With Overlap (Swwan) | Swwan/Advanced/Batch | IMAGE, IMAGE, IMAGE | [nodes/batch.py](../nodes/batch.py#L189) |
| 专用工具 | `SwwanGetLatentRangeFromBatch` | Get Latent Range From Batch (Swwan) | Swwan/Advanced/Batch | LATENT | [nodes/batch.py](../nodes/batch.py#L267) |
| 专用工具 | `InsertLatentToIndex` | Insert Latent To Index (Swwan) | Swwan/Advanced/Batch | LATENT | [nodes/batch.py](../nodes/batch.py#L310) |
| 专用工具 | `SwwanImageBatchFilter` | Image Batch Filter (Swwan) | Swwan/Advanced/Batch | IMAGE, STRING | [nodes/batch.py](../nodes/batch.py#L358) |
| 专用工具 | `SwwanGetImagesFromBatchIndexed` | Get Images From Batch Indexed (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L405) |
| 专用工具 | `SwwanInsertImagesToBatchIndexed` | Insert Images To Batch Indexed (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L436) |
| 专用工具 | `SwwanPadImageBatchInterleaved` | Pad Image Batch Interleaved (Swwan) | Swwan/Advanced/Batch | IMAGE, MASK | [nodes/batch.py](../nodes/batch.py#L496) |
| 专用工具 | `SwwanReplaceImagesInBatch` | Replace Images In Batch (Swwan) | Swwan/Advanced/Batch | IMAGE, MASK | [nodes/batch.py](../nodes/batch.py#L553) |
| 专用工具 | `SwwanReverseImageBatch` | Reverse Image Batch (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L615) |
| 专用工具 | `SwwanImageBatchMulti` | Image Batch Multi (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L636) |
| 专用工具 | `SwwanImageTensorList` | Image Tensor List (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/batch.py](../nodes/batch.py#L670) |
| 专用工具 | `SwwanImageAddMulti` | Image Add Multi (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/batch.py](../nodes/batch.py#L697) |
| 主入口 | `SwwanImageConcatMulti` | Image Concat Multi (Swwan) | Swwan/Image | IMAGE | [nodes/concat.py](../nodes/concat.py#L196) |
| 专用工具 | `SwwanPreviewAnimation` | Preview Animation (Swwan) | Swwan/Advanced/IO | 输出节点 | [nodes/io.py](../nodes/io.py#L318) |
| 兼容入口 | `SwwanImageResizeKJ` | Image Resize (Swwan) · 旧版兼容 | Swwan/Legacy | IMAGE, INT, INT | [nodes/resize.py](../nodes/resize.py#L5) |
| 主入口 | `ImageResizeKJv2Alternative` | Resize Image (Swwan) | Swwan/Image | IMAGE, INT, INT, MASK | [nodes/resize.py](../nodes/resize.py#L82) |
| 主入口 | `ImageResizeByMegapixels` | Image Resize By Megapixels (Swwan) | Swwan/Image | IMAGE, INT, INT, MASK, INT | [nodes/resize.py](../nodes/resize.py#L436) |
| 专用工具 | `SwwanLoadAndResizeImage` | Load And Resize Image (Swwan) | Swwan/Advanced/IO | IMAGE, MASK, INT, INT, STRING | [nodes/io.py](../nodes/io.py#L402) |
| 专用工具 | `SwwanLoadImagesFromFolderKJ` | Load Images From Folder (Swwan) | Swwan/Advanced/IO | IMAGE, MASK, INT, STRING | [nodes/io.py](../nodes/io.py#L555) |
| 专用工具 | `SwwanImageGridtoBatch` | Image Grid to Batch (Swwan) | Swwan/Advanced/Batch | IMAGE | [nodes/transition.py](../nodes/transition.py#L312) |
| 兼容入口 | `SwwanSaveImageKJ` | Save Image (Swwan) · 旧版兼容 | Swwan/Legacy | STRING | [nodes/io.py](../nodes/io.py#L785) |
| 专用工具 | `SwwanSaveStringKJ` | Save String (Swwan) | Swwan/Advanced/IO | STRING | [nodes/io.py](../nodes/io.py#L856) |
| 专用工具 | `SwwanFastPreview` | Fast Preview (Swwan) | Swwan/Advanced/IO | 输出节点 | [nodes/io.py](../nodes/io.py#L906) |
| 专用工具 | `SwwanImageCropByMaskAndResize` | Image Crop By Mask And Resize (Swwan) | Swwan/Advanced/Image | IMAGE, MASK, BBOX | [nodes/mask.py](../nodes/mask.py#L236) |
| 专用工具 | `SwwanImageCropByMask` | Image Crop By Mask (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/mask.py](../nodes/mask.py#L374) |
| 专用工具 | `SwwanImageUncropByMask` | Image Uncrop By Mask (Swwan) | Swwan/Advanced/Mask | IMAGE | [nodes/mask.py](../nodes/mask.py#L420) |
| 专用工具 | `SwwanImageCropByMaskBatch` | Image Crop By Mask Batch (Swwan) | Swwan/Advanced/Batch | IMAGE, MASK | [nodes/mask.py](../nodes/mask.py#L504) |
| 专用工具 | `SwwanImagePadKJ` | Image Pad (Swwan) | Swwan/Advanced/Image | IMAGE, MASK | [nodes/mask.py](../nodes/mask.py#L606) |
| 专用工具 | `SwwanLoadVideosFromFolder` | Load Videos From Folder (Swwan) | Swwan/Advanced/IO | IMAGE | [nodes/io.py](../nodes/io.py#L938) |
| 专用工具 | `SwwanDrawMaskOnImage` | Draw Mask On Image (Swwan) | Swwan/Advanced/Image | IMAGE | [nodes/mask.py](../nodes/mask.py#L777) |
| 专用工具 | `AnySwitch (Swwan)` | Any Switch (Swwan) | Swwan/Advanced/Utils | BOOLEAN, * | [nodes_switch.py](../nodes_switch.py#L16) |
| 主入口 | `AnyBooleanSwitch (Swwan)` | Any Boolean Switch (Swwan) | Swwan/Utils | * | [nodes_switch.py](../nodes_switch.py#L41) |
| 专用工具 | `raiseExceptionOnTrue` | Raise Exception On True (Swwan) | Swwan/Advanced/Utils | BOOLEAN | [nodes_switch.py](../nodes_switch.py#L71) |
| 主入口 | `MathExpression_UTK` | Math Expression (Swwan) | Swwan/Utils | INT, FLOAT, BOOLEAN | [math_expression.py](../math_expression.py#L35) |
| 兼容入口 | `IO_save_image` | IO Save Image (Swwan) · 旧版兼容 | Swwan/Legacy | STRING | [io_nodes.py](../io_nodes.py#L35) |
| 兼容入口 | `IO_save_image_format` | IO Save Image Format (Swwan) · 旧版兼容 | Swwan/Legacy | STRING | [io_nodes.py](../io_nodes.py#L116) |
| 专用工具 | `BoundedImageCrop` | Bounded Image Crop (Swwan) | Swwan/Advanced/Image | IMAGE | [bounded_image_crop.py](../bounded_image_crop.py#L11) |
| 专用工具 | `BoundedImageCropWithMask` | Bounded Image Crop With Mask (Swwan) | Swwan/Advanced/Image | IMAGE, IMAGE_BOUNDS | [bounded_image_crop.py](../bounded_image_crop.py#L56) |
| 主入口 | `RGBA_Safe_Pre` | RGBA Safe Pre (Swwan) | Swwan/RGBA | IMAGE, MASK, BOOLEAN | [rgba_nodes.py](../rgba_nodes.py#L70) |
| 主入口 | `RGBA_Safe_Post` | RGBA Safe Post (Swwan) | Swwan/RGBA | IMAGE, MASK | [rgba_nodes.py](../rgba_nodes.py#L105) |
| 主入口 | `RGBA_Save` | RGBA Save (Swwan) | Swwan/RGBA | 输出节点 | [rgba_nodes.py](../rgba_nodes.py#L139) |
| 兼容入口 | `RGBA_Multi_Save` | RGBA Multi Save (Swwan) · 旧版兼容 | Swwan/Legacy | 输出节点 | [rgba_nodes.py](../rgba_nodes.py#L203) |
| 专用工具 | `SwwanColorShiftFix` | Color Shift Fix (Swwan) | Swwan/Advanced/Image | IMAGE | [color_shift_fix.py](../color_shift_fix.py#L10) |
| 专用工具 | `PatchSageAttentionKJAlternative` | Patch Sage Attention KJ (KJ Alternative) (Swwan) | Swwan/Advanced/Model | MODEL | [sage_attention_patch.py](../sage_attention_patch.py#L276) |
| 专用工具 | `MiniMaxH3MemoryEfficientSageAttentionPatchKJAlternative` | MiniMax H3 Mem Eff Sage Attention Patch (KJ Alternative) (Swwan) | Swwan/Advanced/Model | MODEL | [minimax_h3_sage_attention_patch.py](../minimax_h3_sage_attention_patch.py#L3) |
| 主入口 | `SwwanImageResizeRange` | Image Resize Range (Swwan) | Swwan/Image | IMAGE, MASK | [workflow_tools.py](../workflow_tools.py#L20) |
| 主入口 | `SwwanBlockifyMask` | Blockify Mask (Swwan) | Swwan/Mask | MASK | [workflow_tools.py](../workflow_tools.py#L481) |
| 主入口 | `SwwanImagesToRGB` | Images to RGB (Swwan) | Swwan/Image | IMAGE | [workflow_tools.py](../workflow_tools.py#L571) |
| 主入口 | `SwwanColorConverter` | Color Converter (Swwan) | Swwan/Image | STRING, COLORCODE | [workflow_tools.py](../workflow_tools.py#L354) |
| 主入口 | `SwwanSaveImage` | Save Image (Swwan) | Swwan/IO | STRING | [nodes/save.py](../nodes/save.py#L14) |
| 专用工具 | `SwwanMaskProcess` | Mask Process (Swwan) | Swwan/Advanced/Mask | MASK, MASK | [nodes/mask_tools.py](../nodes/mask_tools.py#L8) |
| 专用工具 | `SwwanMaskCombine` | Mask Combine (Swwan) | Swwan/Advanced/Mask | MASK, INT, INT | [nodes/mask_tools.py](../nodes/mask_tools.py#L72) |
| 专用工具 | `SwwanMaskAnalyze` | Mask Analyze (Swwan) | Swwan/Advanced/Mask | MASK, INT, INT, INT, INT, INT, INT, BOOLEAN | [nodes/mask_tools.py](../nodes/mask_tools.py#L90) |
| 专用工具 | `SwwanMaskSegments` | Mask Segments (Swwan) | Swwan/Advanced/Mask | SEGS, MASK | [nodes/mask_tools.py](../nodes/mask_tools.py#L110) |
| 专用工具 | `SwwanImageMatte` | Image Matte (Swwan) | Swwan/Advanced/Image | IMAGE, IMAGE, MASK | [nodes/mask_tools.py](../nodes/mask_tools.py#L148) |

机器可读接口清单：[node-catalog.json](node-catalog.json)。

接口类型相似不代表行为等价；合并前还需比较批次、空输入、遮罩约定、设备及像素结果。

## 历史 ID 和替代入口

| 历史 ID | 当前独立 ID | 推荐替代 |
| --- | --- | --- |
| LayerUtility: CropByMask V2 | `SwwanCropByMaskV2` | SwwanCropByMaskV5 |
| LayerUtility: CropByMask V3 | `SwwanCropByMaskV3` | SwwanCropByMaskV5 |
| — | `LayerUtility: CropByMask V4` | SwwanCropByMaskV5 |
| LayerUtility: RestoreCropBox | `SwwanRestoreCropBox` | SwwanRestoreCropBoxV4 |
| — | `LayerUtility: RestoreCropBox V2` | SwwanRestoreCropBoxV4 |
| — | `SwwanRestoreCropBoxV3` | SwwanRestoreCropBoxV4 |
| LayerUtility: ImageScaleByAspectRatio V2 | `SwwanImageScaleByAspectRatioV2` | ImageResizeKJv2Alternative |
| Seed (rgthree) | `SwwanSeed` | 保留当前入口 |
| — | `math_calculate` | MathExpression_UTK |
| ImagePass | `SwwanImagePass` | 保留当前入口 |
| ColorMatch | `SwwanColorMatch` | 保留当前入口 |
| SaveImageWithAlpha | `SwwanSaveImageWithAlpha` | SwwanSaveImage |
| ImageConcanate | `SwwanImageConcanate` | SwwanImageConcatMulti |
| ImageConcatFromBatch | `SwwanImageConcatFromBatch` | SwwanImageConcatMulti |
| ImageGridComposite2x2 | `SwwanImageGridComposite2x2` | SwwanImageConcatMulti |
| ImageGridComposite3x3 | `SwwanImageGridComposite3x3` | SwwanImageConcatMulti |
| ImageBatchTestPattern | `SwwanImageBatchTestPattern` | 保留当前入口 |
| ImageGrabPIL | `SwwanImageGrabPIL` | 保留当前入口 |
| WebcamCaptureCV2 | `SwwanWebcamCaptureCV2` | 保留当前入口 |
| AddLabel | `SwwanAddLabel` | 保留当前入口 |
| GetImageSizeAndCount | `SwwanGetImageSizeAndCount` | 保留当前入口 |
| GetLatentSizeAndCount | `SwwanGetLatentSizeAndCount` | 保留当前入口 |
| ImageBatchRepeatInterleaving | `SwwanImageBatchRepeatInterleaving` | 保留当前入口 |
| ImageUpscaleWithModelBatched | `SwwanImageUpscaleWithModelBatched` | 保留当前入口 |
| ImageNormalize_Neg1_To_1 | `SwwanImageNormalize_Neg1_To_1` | 保留当前入口 |
| RemapImageRange | `SwwanRemapImageRange` | 保留当前入口 |
| SplitImageChannels | `SwwanSplitImageChannels` | 保留当前入口 |
| MergeImageChannels | `SwwanMergeImageChannels` | 保留当前入口 |
| ImagePadForOutpaintMasked | `SwwanImagePadForOutpaintMasked` | 保留当前入口 |
| ImagePadForOutpaintTargetSize | `SwwanImagePadForOutpaintTargetSize` | 保留当前入口 |
| ImagePrepForICLora | `SwwanImagePrepForICLora` | 保留当前入口 |
| ImageAndMaskPreview | `SwwanImageAndMaskPreview` | 保留当前入口 |
| CrossFadeImages | `SwwanCrossFadeImages` | 保留当前入口 |
| CrossFadeImagesMulti | `SwwanCrossFadeImagesMulti` | 保留当前入口 |
| TransitionImagesMulti | `SwwanTransitionImagesMulti` | 保留当前入口 |
| TransitionImagesInBatch | `SwwanTransitionImagesInBatch` | 保留当前入口 |
| ImageBatchJoinWithTransition | `SwwanImageBatchJoinWithTransition` | 保留当前入口 |
| ShuffleImageBatch | `SwwanShuffleImageBatch` | 保留当前入口 |
| GetImageRangeFromBatch | `SwwanGetImageRangeFromBatch` | 保留当前入口 |
| ImageBatchExtendWithOverlap | `SwwanImageBatchExtendWithOverlap` | 保留当前入口 |
| GetLatentRangeFromBatch | `SwwanGetLatentRangeFromBatch` | 保留当前入口 |
| ImageBatchFilter | `SwwanImageBatchFilter` | 保留当前入口 |
| GetImagesFromBatchIndexed | `SwwanGetImagesFromBatchIndexed` | 保留当前入口 |
| InsertImagesToBatchIndexed | `SwwanInsertImagesToBatchIndexed` | 保留当前入口 |
| PadImageBatchInterleaved | `SwwanPadImageBatchInterleaved` | 保留当前入口 |
| ReplaceImagesInBatch | `SwwanReplaceImagesInBatch` | 保留当前入口 |
| ReverseImageBatch | `SwwanReverseImageBatch` | 保留当前入口 |
| ImageBatchMulti | `SwwanImageBatchMulti` | 保留当前入口 |
| ImageTensorList | `SwwanImageTensorList` | 保留当前入口 |
| ImageAddMulti | `SwwanImageAddMulti` | 保留当前入口 |
| ImageConcatMulti | `SwwanImageConcatMulti` | 保留当前入口 |
| PreviewAnimation | `SwwanPreviewAnimation` | 保留当前入口 |
| ImageResizeKJ | `SwwanImageResizeKJ` | ImageResizeKJv2Alternative |
| LoadAndResizeImage | `SwwanLoadAndResizeImage` | 保留当前入口 |
| LoadImagesFromFolderKJ | `SwwanLoadImagesFromFolderKJ` | 保留当前入口 |
| ImageGridtoBatch | `SwwanImageGridtoBatch` | 保留当前入口 |
| SaveImageKJ | `SwwanSaveImageKJ` | SwwanSaveImage |
| SaveStringKJ | `SwwanSaveStringKJ` | 保留当前入口 |
| FastPreview | `SwwanFastPreview` | 保留当前入口 |
| ImageCropByMaskAndResize | `SwwanImageCropByMaskAndResize` | 保留当前入口 |
| ImageCropByMask | `SwwanImageCropByMask` | 保留当前入口 |
| ImageUncropByMask | `SwwanImageUncropByMask` | 保留当前入口 |
| ImageCropByMaskBatch | `SwwanImageCropByMaskBatch` | 保留当前入口 |
| ImagePadKJ | `SwwanImagePadKJ` | 保留当前入口 |
| LoadVideosFromFolder | `SwwanLoadVideosFromFolder` | 保留当前入口 |
| — | `IO_save_image` | SwwanSaveImage |
| — | `IO_save_image_format` | SwwanSaveImage |
| — | `RGBA_Multi_Save` | SwwanSaveImage |
