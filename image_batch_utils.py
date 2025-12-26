
import torch
import comfy.utils

class ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "images": ("IMAGE", ),
                      }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "Swwan/Image"

    def doit(self, images):
        if len(images) == 0:
            return ()
        if len(images) == 1:
            img = images[0]
            if img.ndim == 3:  # add batch dim if missing
                img = img.unsqueeze(0)
            return (img,)

        # Start with the first image
        image1 = images[0]
        if image1.ndim == 3:
            image1 = image1.unsqueeze(0)

        for image2 in images[1:]:
            # Ensure batch dim
            if image2.ndim == 3:
                image2 = image2.unsqueeze(0)

            # Ensure same device
            if image2.device != image1.device:
                image2 = image2.to(image1.device)

            # Ensure HxW match exactly
            H, W = image1.shape[1], image1.shape[2]
            if image2.shape[1] != H or image2.shape[2] != W:
                image2 = comfy.utils.common_upscale(
                    image2.movedim(-1, 1),  # move channels first
                    W,  # width
                    H,  # height
                    "lanczos",
                    "center"
                ).movedim(1, -1)  # move channels back last

            # Ensure channels match
            if image2.shape[3] != image1.shape[3]:
                # simple fix: truncate or pad channels
                min_C = min(image1.shape[3], image2.shape[3])
                image1 = image1[:, :, :, :min_C]
                image2 = image2[:, :, :, :min_C]

            # Concatenate along batch dimension
            image1 = torch.cat((image1, image2), dim=0)

        return (image1,)


class ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "Swwan/Image"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )

NODE_CLASS_MAPPINGS = {
    "SwwanImageListToImageBatch": ImageListToImageBatch,
    "SwwanImageBatchToImageList": ImageBatchToImageList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwwanImageListToImageBatch": "Image List to Image Batch",
    "SwwanImageBatchToImageList": "Image Batch to Image List",
}
