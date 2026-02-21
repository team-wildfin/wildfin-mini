from .defaults import DEFAULT_FIELDS
from vision_bench.typing.experiment import Experiment
from vision_bench.typing.types import UniformConfig, FocalLossConfig
from .cvpr import CVPR_EXPS

DINOV3_BASE_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_base_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dinov3_base",
            'pooling': 'attention',        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["frames_w_temp"]
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

DINOV3_BASE_MEAN = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_base_mean_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dinov3_base",
            'pooling': 'mean',        
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': False,
            'freeze_backbone': False       
        }
    )
    for sliding_style in ["frames_w_temp"]
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_LARGE_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'videomae_large_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "videomae_large",
            'pooling': 'attention',        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["sliding_window_w_temp"]
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_LARGE_MEAN = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'videomae_large_mean_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "videomae_large",
            'pooling': 'mean',        
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': False,
            'freeze_backbone': False       
        }
    )
    for sliding_style in ["sliding_window_w_temp"]
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

ECCV_EXPS = (
    DINOV3_BASE_ATTENTION + 
    DINOV3_BASE_MEAN +
    VIDEOMAE_LARGE_ATTENTION + 
    VIDEOMAE_LARGE_MEAN
)

ECCV_CORALCAM = [exp for exp in ECCV_EXPS if exp.dataset == "coralcam"]
ECCV_FISHFOLLOW = [exp for exp in ECCV_EXPS if exp.dataset == "fishfollow"]
ECCV_FISHFOLLOW_VIDEOMAE_LARGE = [exp for exp in ECCV_FISHFOLLOW if exp.backbone == "videomae_large"]
ECCV_FISHFOLLOW_DINOV3_BASE = [exp for exp in ECCV_FISHFOLLOW if exp.backbone == "dinov3_base"]