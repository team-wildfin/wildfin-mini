print("experiment defaults importing...")
from .defaults import DEFAULT_FIELDS
print("experiment types importing...")
from vision_bench.typing.experiment import Experiment
print("loss types importing...")
from vision_bench.typing.types import UniformConfig, FocalLossConfig
print("experiment constants importing...")
from .neurips import DINO_WEIGHTED_EXPS, VIDEOMAE_WEIGHTED_EXPS, RESNET50_WEIGHTED_EXPS, RESNET_FULLTUNE
print("experiment packages imported")

DINOV3_LARGE_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_large_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dinov3_large",
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

DINOV3_LARGE_MEAN = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_large_mean_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dinov3_large",
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

VJEPA2_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'vjepa2_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "vjepa2",
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

VJEPA2_MEAN = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'vjepa2_mean_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "vjepa2",
            'pooling': 'mean',        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': False,
            'freeze_backbone': False        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["sliding_window_w_temp"]
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'videomae_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "videomae",
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

RESNET_ATTENTION = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'resnet50_attention_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "resnet50",
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

CVPR_EXPS = (DINOV3_LARGE_ATTENTION + 
             DINOV3_LARGE_MEAN +
             VJEPA2_ATTENTION + 
             VJEPA2_MEAN +
             RESNET_ATTENTION + 
             VIDEOMAE_WEIGHTED_EXPS + 
             VIDEOMAE_ATTENTION +
             RESNET50_WEIGHTED_EXPS + 
             RESNET_FULLTUNE
            ) 