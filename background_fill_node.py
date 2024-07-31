# a comfyUI node to fill a background given an image and a mask
# The node should have the following inputs:
# - image: the input image
# - mask: the mask of the object to segment
import torch

class BackgroundFillNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ('IMAGE', {}),
                "mask": ("MASK",),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")


    # image is a 4D tensor of shape (batch_size, channels, height, width)
    # mask is a 3D tensor of shape (batch_size, height, width)
    def main(self, image, mask):

        # print("image", image)
        # print("mask", mask)

        # where mask is 0, make the image white
        image[mask == 0] = 255

        return (image, mask,)
    
class CombineMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    # mask is a 3D tensor of shape (batch_size, height, width)
    def main(self, mask1, mask2):

        print("mask1", mask1.shape)
        print("mask2", mask2.shape)

        # where mask is 0, make the image white
        # mark it as zero if mask1 is 1 and mask2 is 1
        # mark it zero is mask1 is 0 and mask2 is 0 or 1
        mask = mask1 + mask2

        return (mask,)

NODE_CLASS_MAPPINGS = {
    "Background Fill Node": BackgroundFillNode,
    "Combine Mask Node": CombineMaskNode
}