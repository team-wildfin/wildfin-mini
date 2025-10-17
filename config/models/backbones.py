from config.models.models import ModelConfig, BackBoneConfig
from typing import Dict
from fish_benchmark.typing.types import SlidingStyle
from ..sliding_styles import *
from typing import List, Dict, Callable, Literal
import torch.nn as nn
from transformers import AutoModel
import torch
from fish_benchmark.data.preprocessors import TorchVisionPreprocessor
from transformers import VideoMAEModel, ResNetModel


VIDEOMAE = BackBoneConfig(
    name="videomae",
    architecture="transformer",
    hidden_size=768,
    input_ndim=4,
    output_ndim=2,
) 
DINO = BackBoneConfig(
    name="dino",
    architecture="transformer",
    hidden_size=768,
    input_ndim=3,
    output_ndim=2
)
DINO_LARGE = BackBoneConfig(
    name="dino_large",
    architecture="transformer",
    hidden_size=768,
    input_ndim=3,
    output_ndim=2,
)
DINOV3_BASE = BackBoneConfig(
    name="dinov3_base",
    architecture="transformer",
    hidden_size=768,
    input_ndim=3,
    output_ndim=2
)

DINOV3_LARGE = BackBoneConfig(
    name="dinov3_large",
    architecture="transformer",
    hidden_size=1024,
    input_ndim=3,
    output_ndim=2
)



RESNET50 = BackBoneConfig(
    name="resnet50",
    architecture="cnn",
    hidden_size=2048,
    input_ndim=3,
    output_ndim=2,
)

VJEPA2 = BackBoneConfig(
    name="vjepa2",
    architecture="transformer",
    hidden_size=1024,
    input_ndim=4,
    output_ndim=2,
)

BACKBONE_CONFIGS: Dict[str, BackBoneConfig] = {
    "videomae": VIDEOMAE,
    "dino": DINO,
    "dino_large": DINO_LARGE,
    "dinov3_base": DINOV3_BASE,
    "dinov3_large": DINOV3_LARGE,
    "resnet50": RESNET50,
    "vjepa2": VJEPA2,
}

BACKBONE_MODULES: Dict[str, Callable[[], nn.Module]] = {
    "vjepa2": lambda: AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        device_map="auto",
        attn_implementation="sdpa"
    ), 
    "videomae": lambda: VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"), 
    "dino": lambda: AutoModel.from_pretrained('facebook/dinov2-base'),
    "dino_large": lambda: AutoModel.from_pretrained('facebook/dinov2-large'),
    "dinov3_base": lambda: AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "dinov3_large": lambda: AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m"),
    "resnet50": lambda: ResNetModel.from_pretrained('microsoft/resnet-50'),
}

PREPROCESSORS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "vjepa2": TorchVisionPreprocessor(crop_size=(256, 256), resize_shortest=256),
    "videomae": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dino_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3_base": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "dinov3_large": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
    "resnet50": TorchVisionPreprocessor(crop_size=(224, 224), resize_shortest=256),
}

MODEL_SLIDING_STYLES: Dict[str, List[SlidingStyle]] = {
    "vjepa2": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "videomae": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "dino": [FRAMES_W_TEMP, TEST_FRAMES],
    "dino_large": [FRAMES_W_TEMP, TEST_FRAMES],
    "dinov3_base": [FRAMES_W_TEMP, TEST_FRAMES],
    "dinov3_large": [FRAMES_W_TEMP, TEST_FRAMES],
    "resnet50": [FRAMES_W_TEMP, TEST_FRAMES],
}