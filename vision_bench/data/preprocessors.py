from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torch


class TorchVisionPreprocessor:
    def __init__(self, crop_size=(224, 224), resize_shortest=256,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 interpolation=InterpolationMode.BICUBIC):
        
        self.resize = v2.Resize(resize_shortest, interpolation=interpolation, antialias=False)
        self.crop = v2.CenterCrop(crop_size)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def _process_single(self, img: torch.Tensor) -> torch.Tensor:
        img = img / 255.0
        img = self.resize(img)
        img = self.crop(img)
        img = (img - self.mean) / self.std
        return img

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim == 3:
            # Single image [3, H, W]
            return self._process_single(image_tensor)
        elif image_tensor.ndim == 4:
            # Batch of images [B, 3, H, W]
            return torch.stack([self._process_single(img) for img in image_tensor], dim=0)
        else:
            raise ValueError(f"Expected image tensor of shape [3, H, W] or [B, 3, H, W], got {image_tensor.shape}")
