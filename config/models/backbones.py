from config.models.models import BackBoneConfig
from typing import Dict
from config.data.sliding_styles import *
from typing import Dict

print("defining backbone configs...")
VIDEOMAE = BackBoneConfig(
    name="videomae",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=768,
    input_ndim=4,
    output_ndim=2,
) 
VIDEOMAE_LARGE = BackBoneConfig(
    name="videomae_large",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=1024,
    input_ndim=4,
    output_ndim=2,
)

DINO = BackBoneConfig(
    name="dino",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=768,
    input_ndim=3,
    output_ndim=2,
)
DINO_LARGE = BackBoneConfig(
    name="dino_large",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=768,
    input_ndim=3,
    output_ndim=2,
)
DINOV3_BASE = BackBoneConfig(
    name="dinov3_base",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=768,
    input_ndim=3,
    output_ndim=2
)

DINOV3_LARGE = BackBoneConfig(
    name="dinov3_large",
    architecture="transformer",
    crop_size=(224, 224),
    hidden_size=1024,
    input_ndim=3,
    output_ndim=2
)

RESNET50 = BackBoneConfig(
    name="resnet50",
    architecture="cnn",
    crop_size=(224, 224),
    hidden_size=2048,
    input_ndim=3,
    output_ndim=2,
)

VJEPA2 = BackBoneConfig(
    name="vjepa2",
    architecture="transformer",
    crop_size=(256, 256),
    hidden_size=1024,
    input_ndim=4,
    output_ndim=2,
)


BACKBONE_CONFIGS: Dict[str, BackBoneConfig] = {
    backbone.name: backbone
    for backbone in globals().values()
    if isinstance(backbone, BackBoneConfig)
}