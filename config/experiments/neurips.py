from fish_benchmark.typing.types import (
    WeightConfig,
    InverseConfig,
    UniformConfig,
    FocalLossConfig,
)
from fish_benchmark.typing.experiment import Experiment
from typing import List, Dict, Any
from defaults import DEFAULT_FIELDS



DINO_ATTENTION_BALANCED_UNIFORM = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f"dino_attention_uniform_{dataset}_{sliding_style}",
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "dino_large",
            'pooling': 'attention',        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': UniformConfig(),
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True        # overrides DEFAULT_FIELDS
        }
    )
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

VJEPA_BALANCED_UNIFORM = [
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f"vjepa2_balanced_uniform_{dataset}_{sliding_style}",
            'dataset': dataset,
            'sliding_style': sliding_style,
            'backbone': "vjepa2",
            'sampler': "balanced",
            'weight_config': UniformConfig(),
            'epochs': 40,
            'fulltune': True,
            'freeze_backbone': True,  
        }               # overrides DEFAULT_FIELDS
        )
    for sliding_style in ["sliding_window_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_BALANCED_UNIFORM = [
    Experiment(
        **dict(
            **DEFAULT_FIELDS,
            id=f"videomae_balanced_uniform_{dataset}_{sliding_style}",
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="balanced",
            weight_config=UniformConfig(),
            epochs=40,
            fulltune=False,
        )
    )
    for sliding_style in ["sliding_window_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_RANDOM_UNIFORM = [
    Experiment(
        **dict(
            id=f"videomae_random_uniform_{dataset}_{sliding_style}",
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="videomae",
            sampler="random",
            weight_config=UniformConfig(),
            epochs=100,
            fulltune=False,
            **DEFAULT_FIELDS,
        )
    )
    for sliding_style in ["sliding_window_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_BALANCED_FOCAL = [
    Experiment(
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
    for sliding_style in ["sliding_window_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_BALANCED_FOCAL_2 = [
    Experiment(
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
    for sliding_style in ["sliding_window_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    # Best f1 score
]

VIDEOMAE_RANDOM_FOCAL = [
    Experiment(
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
    for sliding_style in ['sliding_window_w_temp']
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_RANDOM_INVERSE = [
    Experiment(
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
    for sliding_style in ['sliding_window_w_temp']
    for dataset in ["coralcam", "fishfollow"]
]

VIDEOMAE_WEIGHTED_EXPS: List[Experiment] = (
    VIDEOMAE_BALANCED_FOCAL
    + VIDEOMAE_BALANCED_FOCAL_2
    + VIDEOMAE_RANDOM_FOCAL
    + VIDEOMAE_RANDOM_INVERSE
    + VIDEOMAE_BALANCED_UNIFORM
    + VIDEOMAE_RANDOM_UNIFORM
)

DINO_RANDOM_UNIFORM = [
    Experiment(
        **dict(
            id=f"dino_random_uniform_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="random",
            weight_config=UniformConfig(),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True, False]
]

DINO_BALANCED_UNIFORM = [
    Experiment(
        **dict(
            id=f"dino_balanced_uniform_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="balanced",
            weight_config=UniformConfig(),
            epochs=40,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [False]
]

DINO_BALANCED_UNIFORM_FINETUNE = [
    Experiment(
        **dict(
            id=f"dino_balanced_uniform_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="balanced",
            weight_config=UniformConfig(),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True]
]

DINO_BALANCED_FOCAL = [
    Experiment(
        **dict(
            id=f"dino_balanced_focal_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=40,
            fulltune=fulltune,
        )
    )
    for fulltune in [True, False]
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_BALANCED_FOCAL_FINETUNE = [
    Experiment(
        **dict(
            id=f"dino_balanced_focal_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="balanced",
            weight_config=FocalLossConfig(focal_loss_gamma=5.0, focal_loss_alpha=0.75),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for fulltune in [True]
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_RANDOM_FOCAL = [
    Experiment(
        **dict(
            id=f"dino_random_focal_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="random",
            weight_config=FocalLossConfig(focal_loss_gamma=1.0, focal_loss_alpha=0.5),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for fulltune in [True, False]
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_RANDOM_INVERSE = [
    Experiment(
        **dict(
            id=f"dino_random_inverse_{dataset}_{sliding_style}_{fulltune}",
            **DEFAULT_FIELDS,
            dataset=dataset,
            sliding_style=sliding_style,
            backbone="dino_large",
            sampler="random",
            weight_config=InverseConfig(),
            epochs=100,
            fulltune=fulltune,
        )
    )
    for fulltune in [True, False]
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
]

DINO_WEIGHTED_EXPS: List[Experiment] = (
    DINO_BALANCED_FOCAL 
    + DINO_BALANCED_UNIFORM_FINETUNE
    + DINO_RANDOM_FOCAL 
    + DINO_RANDOM_INVERSE 
    + DINO_BALANCED_UNIFORM 
    + DINO_RANDOM_UNIFORM
)

RESNET_BALANCED_UNIFORM = [
    Experiment(
        **dict(
            id=f"resnet_uniform_{dataset}_{sliding_style}_{fulltune}",
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
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True, False]
]

RESNET_RANDOM_UNIFORM = [
    Experiment(
        **dict(
            id=f"resnet_random_uniform_{dataset}_{sliding_style}_{fulltune}",
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
    for sliding_style in ["frames_w_temp"]
    for dataset in ["coralcam", "fishfollow"]
    for fulltune in [True, False]
]

RESNET_RANDOM_INVERSE = [
    Experiment(
        **dict(
            id=f"resnet_random_inverse_{dataset}_{sliding_style}_{fulltune}",
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
    for sliding_style in ["frames_w_temp"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET_RANDOM_FOCAL = [
    Experiment(
        **dict(
            id=f"resnet_random_focal_{dataset}_{sliding_style}_{fulltune}",
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
    for sliding_style in ["frames_w_temp"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET_BALANCED_FOCAL = [
    Experiment(
        **dict(
            id=f"resnet_balanced_focal_{dataset}_{sliding_style}_{fulltune}",
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
    for sliding_style in ["frames_w_temp"]
    for fulltune in [True, False]
    for dataset in ["coralcam", "fishfollow"]
]

RESNET50_WEIGHTED_EXPS: List[Experiment] = (
    RESNET_BALANCED_UNIFORM
    + RESNET_RANDOM_UNIFORM
    + RESNET_RANDOM_FOCAL 
    + RESNET_BALANCED_FOCAL 
    + RESNET_RANDOM_INVERSE
)