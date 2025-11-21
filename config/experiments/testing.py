from vision_bench.typing.types import (
    WeightConfig,
    InverseConfig,
    UniformConfig,
    FocalLossConfig,
)
from vision_bench.typing.experiment import Experiment
from typing import List, Dict, Any
from .defaults import DEFAULT_FIELDS

DINOV3_BASE_MEAN = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_base_mean_uniform_coralcam_frames_w_temp',
            'dataset': 'coralcam',
            'sliding_style': 'frames_w_temp',
            'backbone': "dinov3_base",
            'pooling': 'mean',        
            'sampler': "balanced",
            'weight_config': UniformConfig(),
            'epochs': 40,
            'fulltune': False,
            'freeze_backbone': False       
        }
    )
]