import torch.nn as nn
from transformers import AutoModel
import torch
from transformers import VideoMAEModel, ResNetModel
from typing import Dict, Callable

BACKBONE_MODULES: Dict[str, Callable[[], nn.Module]] = {
    "vjepa2": lambda: AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        device_map="auto",
        attn_implementation="sdpa"
    ), 
    "videomae": lambda: VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"), 
    "videomae_large": lambda: VideoMAEModel.from_pretrained("MCG-NJU/videomae-large"),
    "dino": lambda: AutoModel.from_pretrained('facebook/dinov2-base'),
    "dino_large": lambda: AutoModel.from_pretrained('facebook/dinov2-large'),
    "dinov3_base": lambda: AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "dinov3_large": lambda: AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m"),
    "resnet50": lambda: ResNetModel.from_pretrained('microsoft/resnet-50'),
}
