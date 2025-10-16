from defaults import DEFAULT_FIELDS
from fish_benchmark.typing.experiment import Experiment
from fish_benchmark.typing.types import UniformConfig, FocalLossConfig
from neurips import DINO_WEIGHTED_EXPS, VIDEOMAE_WEIGHTED_EXPS, RESNET50_WEIGHTED_EXPS

DINOV3 = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'dinov3_{pooling}_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dinov3",
            'pooling': pooling,        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["frames_w_temp"]
    for pooling in ['attention', 'mean']
    for weight_config in [UniformConfig(), FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0)]
    for dataset in ["coralcam", "fishfollow"]
]

VJEPA2 = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'vijepa2_{pooling}_{weight_config.weight_method}_{dataset}_{sliding_style}',
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "vjepa2",
            'pooling': pooling,        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': weight_config,
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["sliding_window_w_temp"]
    for pooling in ['attention', 'mean']
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

CVPR_EXPS = DINOV3 + VJEPA2 + RESNET_ATTENTION + DINO_WEIGHTED_EXPS + VIDEOMAE_WEIGHTED_EXPS + RESNET50_WEIGHTED_EXPS