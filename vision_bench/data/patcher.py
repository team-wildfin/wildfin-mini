import numpy as np
import math

class Patcher: 
    def __init__(self, patch_type, h, w):
        assert patch_type in ["absolute", "relative"], f"patch_type {patch_type} not recognized"
        # if patch_type == "absolute": h is the height of each patch, w is the width of each patch
        # if patch_type == "relative": the full height of the image would be cut into h patches, and the full width of the image would be cut into w patches
        self.patch_type = patch_type
        self.h = h
        self.w = w
    
    def pad_to_multiple_np(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Pads a HWC (Height-Width-Channel) image so that its height is a multiple of h
        and width is a multiple of w using 0-padding.

        Args:
            image (np.ndarray): Input image of shape (H, W, C)
            h (int): Desired height multiple
            w (int): Desired width multiple

        Returns:
            np.ndarray: Padded image
        """
        H, W, C = image.shape
        pad_h = (h - H % h) % h
        pad_w = (w - W % w) % w
        padding = ((0, pad_h), (0, pad_w), (0, 0))  # Pad bottom and right
        return np.pad(image, padding, mode='constant', constant_values=0)

    def patch(self, img, h, w):
        img = self.pad_to_multiple_np(img, h, w)
        return img.reshape(-1, h, w, img.shape[2])
    
    def __call__(self, img):
        if self.patch_type == "absolute":
            return self.patch(img, self.h, self.w)
        elif self.patch_type == "relative":
            # Split the image into patches
            h = math.ceil(img.shape[0] / self.h) #h is the 
            w = math.ceil(img.shape[1] / self.w)
            return self.patch(img, h, w)