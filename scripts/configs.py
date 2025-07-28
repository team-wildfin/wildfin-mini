from fish_benchmark.types import (
    TrainConfig,
    WeightConfig,
    InverseConfig,
    UniformConfig,
    FocalLossConfig,
)
from typing import List, Dict, Any

DEFAULT_FIELDS = {
    'pooling': 'mean',
    'classifier': 'mlp',
    'monitor': 'val_mAP',
    'learning_rate': 0.00005,
    'batch_size': 32,
    'weight_decay': 0.001,
    'shuffle': False,
    'monitor': 'val_mAP',
    'optimizer': 'adam',
    'label_type': 'onehot',
    'max_samples_per_class': 1000,
}

RESNET_BALANCED_UNIFORM = [
    TrainConfig(
        **dict(
            id=f"resnet_uniform_{dataset}_{sliding_style}",
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="resnet50",
            sampler="balanced",
            weight_config=UniformConfig(),
            epochs=40,
            fulltune = fulltune, 
            **DEFAULT_FIELDS,
        )
    )
    for sliding_style in ["frames"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True, False]
]

RESNET_RANDOM_UNIFORM = [
    TrainConfig(
        **dict(
            id=f"resnet_random_uniform_{dataset}_{sliding_style}",
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="resnet50",
            sampler="random",
            weight_config=UniformConfig(),
            epochs=100,
            fulltune = fulltune, 
            **DEFAULT_FIELDS,
        )
    )
    for sliding_style in ["frames"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True, False]
]

VIDEOMAE_BALANCED_FOCAL = [
    TrainConfig(
        **dict(
            id=f"videomae_balanced_focal_{dataset}_{sliding_style}",
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=40,
            fulltune=False,
            **DEFAULT_FIELDS,
        )
    )
    for sliding_style in ["sliding_window_w_stride"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_BALANCED_FOCAL_2 = [
    TrainConfig(
        **dict(
            id=f"videomae_balanced_focal_2_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=5.0, focal_loss_alpha=0.75),
            epochs=40,
            fulltune=False,
        )
    )
    for sliding_style in ["sliding_window_w_stride"]
    for dataset in ["coralcam", "fishfollow"]
    # Best f1 score
]

VIDEOMAE_RANDOM_FOCAL = [
    TrainConfig(
        **dict(
            id=f"videomae_random_focal_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="random",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=100,
            fulltune=False,
        )
    )
    for sliding_style in ["sliding_window_w_stride"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_RANDOM_INVERSE = [
    TrainConfig(
        **dict(
            id=f"videomae_random_inverse_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="random",
            weight_config=InverseConfig(),
            epochs=100,
            fulltune=False,
        )
    )
    for sliding_style in ["sliding_window_w_stride"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_WEIGHTED_EXPS: List[TrainConfig] = (
    VIDEOMAE_BALANCED_FOCAL
    + VIDEOMAE_BALANCED_FOCAL_2
    + VIDEOMAE_RANDOM_FOCAL
    + VIDEOMAE_RANDOM_INVERSE
)

DINO_BALANCED_FOCAL = [
    TrainConfig(
        **dict(
            id=f"dino_balanced_focal_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=40,
            fulltune=False,
        )
    )
    for sliding_style in ["frames"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_RANDOM_FOCAL = [
    TrainConfig(
        **dict(
            id=f"dino_random_focal_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="random",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=100,
            fulltune=False,
        )
    )
    for sliding_style in ["frames"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_RANDOM_INVERSE = [
    TrainConfig(
        **dict(
            id=f"dino_random_inverse_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="random",
            weight_config=InverseConfig(),
            epochs=100,
            fulltune=False,
        )
    )
    for sliding_style in ["frames"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_WEIGHTED_EXPS: List[TrainConfig] = (
    DINO_BALANCED_FOCAL + DINO_RANDOM_FOCAL + DINO_RANDOM_INVERSE
)


RESNET_RANDOM_INVERSE = [
    TrainConfig(
        **dict(
            id=f"resnet_random_inverse_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="resnet50",
            sampler="random",
            weight_config=InverseConfig(),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET_RANDOM_FOCAL = [
    TrainConfig(
        **dict(
            id=f"resnet_random_focal_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="resnet50",
            sampler="random",
            weight_config=FocalLossConfig(focal_loss_gamma=5.0, focal_loss_alpha=0.75),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET_BALANCED_FOCAL = [
    TrainConfig(
        **dict(
            id=f"resnet_balanced_focal_{dataset}_{sliding_style}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="resnet50",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=40,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET50_WEIGHTED_EXPS: List[TrainConfig] = (
    RESNET_RANDOM_FOCAL + RESNET_BALANCED_FOCAL + RESNET_RANDOM_INVERSE
)
