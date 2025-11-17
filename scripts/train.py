import os
import subprocess
import yaml
import os
from config.experiments.neurips import *
from typing import List, Optional, Callable
from fish_benchmark.typing.experiment import Experiment
from fish_benchmark.management.wandb_matcher import WandbRunMatcher
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.utils.submission import get_slurm_submission_command
from config.experiments.cvpr import CVPR_EXPS
import pprint
from fish_benchmark.management.query import query_pending_experiments
from fish_benchmark.execution.trainer import Trainer

PARALLEL = False
OUTPUT_BASE = os.path.join("logs", "train")
ENTITY = "fish-benchmark"
TRAINING_PROJECT = "fishfollow"
def filt(exp: Experiment) -> bool:
    return (exp.backbone in ['dinov3_large', 'vjepa2', 'resnet50', 'videomae'] and 
            exp.pooling in ['attention', 'mean'] and 
            exp.weight_config.weight_method in ['uniform', 'focal_loss'] and 
            exp.dataset == TRAINING_PROJECT) 

CONFIG_OUT = "generated_configs"
os.makedirs(CONFIG_OUT, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
logger = setup_logger(
    "train", os.path.join(OUTPUT_BASE, "train.log"), console=True, file=True
)


if __name__ == "__main__":
    trainer = Trainer(
        [exp for exp in CVPR_EXPS if filt(exp)], 
        WandbRunMatcher(ENTITY, TRAINING_PROJECT), 
        parallel=PARALLEL
    )
    trainer.run()