from pydantic import BaseModel
from fish_benchmark.types import SlidingStyle
from sliding_styles import *
from typing import List, Dict, Callable, Literal
import torch.nn as nn
from transformers import AutoModel
import torch
from fish_benchmark.data.preprocessors import TorchVisionPreprocessor
from transformers import VideoMAEModel, ResNetModel

class ModelConfig(BaseModel):
    name: str
    category: Literal['cnn', 'transformer']
    input_ndim: int
    output_ndim: int

VIDEOMAE = ModelConfig(
    name="videomae",
    category="transformer",
    input_ndim=4,
    output_ndim=2,
) 
DINO = ModelConfig(
    name="dino",
    category="transformer",
    input_ndim=3,
    output_ndim=2
)
DINO_LARGE = ModelConfig(
    name="dino_large",
    category="transformer",
    input_ndim=3,
    output_ndim=2,
)
DINO_V3 = ModelConfig(
    name="dinov3",
    category="transformer",
    input_ndim=3,
    output_ndim=2
)
DINO_V3_LARGE = ModelConfig(
    name="dinov3_large",
    category="transformer",
    input_ndim=3,
    output_ndim=2
)
RESNET50 = ModelConfig(
    name="resnet50",
    category="cnn",
    input_ndim=3,
    output_ndim=2,
)

VJEPA2 = ModelConfig(
    name="vjepa2",
    category="transformer",
    input_ndim=3,
    output_ndim=2,
)

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "videomae": VIDEOMAE,
    "dino": DINO,
    "dino_large": DINO_LARGE,
    "dinov3": DINO_V3,
    "dinov3_large": DINO_V3_LARGE,
    "resnet50": RESNET50,
    "vjepa2": VJEPA2,
}

MODULES: Dict[str, nn.Module] = {
    "vjepa2": AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    ), 
    "videomae": VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"), 
    "dino": AutoModel.from_pretrained('facebook/dinov2-base'),
    "dino_large": AutoModel.from_pretrained('facebook/dinov2-large'),
    "dinov3": AutoModel.from_pretrained('facebook/dinov3-base'),
    "dinov3_large": AutoModel.from_pretrained('facebook/dinov3-large'),
    "resnet50": ResNetModel.from_pretrained('microsoft/resnet-50'),
}

PREPROCESSORS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "vjepa2": TorchVisionPreprocessor(crop_size=(256, 256), resize_shortest=256),
    "videomae": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "resnet50": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
}

MODEL_SLIDING_STYLES: Dict[str, List[SlidingStyle]] = {
    "vjepa2": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "videomae": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "dino": [FRAMES, TEST_FRAMES],
    "dino_large": [FRAMES, TEST_FRAMES],
    "dinov3": [FRAMES, TEST_FRAMES],
    "dinov3_large": [FRAMES, TEST_FRAMES],
    "resnet50": [FRAMES, TEST_FRAMES],
}