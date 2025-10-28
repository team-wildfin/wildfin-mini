from fish_benchmark.typing.experiment import Experiment, Evaluation
from config.experiments.defaults import DEFAULT_FIELDS
from fish_benchmark.typing.types import UniformConfig, FocalLossConfig
import wandb
import pprint

EXP = (
    Experiment(
        **{
            **DEFAULT_FIELDS,
            'id': f'vjepa2_mean_focal_loss_coralcam_sliding_window_w_temp',
            'dataset': "coralcam",
            'sliding_style': "sliding_window_w_temp",
            'backbone': "vjepa2",
            'pooling': 'mean',        # overrides DEFAULT_FIELDS
            'sampler': "balanced",
            'weight_config': FocalLossConfig(focal_loss_alpha=0.75, focal_loss_gamma=5.0),
            'epochs': 40,
            'fulltune': False,
            'freeze_backbone': False        # overrides DEFAULT_FIELDS
        }
    )
)
training_run = wandb.Api().run(f"fish-benchmark/fishfollow/mc08kaid")
train_config = Experiment.model_validate(training_run.config)
pprint.pprint(train_config)
EVAL = Evaluation.model_validate(
    train_config.model_dump() |
    {
        "test_sliding_style": "test_sliding_window",
        "training_run_id": "sample",
        "training_entity": "sample",
        "training_project": "sample"
    }
)

print(type(EVAL.weight_config))
