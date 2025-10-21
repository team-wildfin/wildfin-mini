from fish_benchmark.data.preprocessors import TorchVisionPreprocessor
from typing import Dict, Callable, Any
PREPROCESSORS: Dict[str, Callable[[Any], Any]] = {
    "vjepa2": TorchVisionPreprocessor(crop_size=(256, 256), resize_shortest=256),
    "videomae": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3_base": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "resnet50": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
}