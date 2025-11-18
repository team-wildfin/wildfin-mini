import os
from config.experiments.neurips import *
from fish_benchmark.typing.experiment import Experiment
from fish_benchmark.management.wandb_matcher import WandbRunMatcher
from fish_benchmark.utils.general import setup_logger
from config.experiments.cvpr import CVPR_EXPS
from fish_benchmark.execution.trainer import Trainer
from config.experiments.defaults import MISSING_VALUES

PARALLEL = False
OUTPUT_BASE = os.path.join("logs", "train")
ENTITY = "fish-benchmark"
TRAINING_PROJECT = "fishfollow"
def filt(exp: Experiment) -> bool:
    return (exp.backbone in ['dinov3_large', 'vjepa2', 'resnet50', 'videomae'] and 
            exp.pooling in ['attention', 'mean'] and 
            exp.weight_config.weight_method in ['uniform', 'focal_loss'] and 
            exp.dataset == TRAINING_PROJECT) 
ALL_EXPS = [exp for exp in CVPR_EXPS if filt(exp)]
CONFIG_OUT = "generated_configs"
os.makedirs(CONFIG_OUT, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
logger = setup_logger(
    "train", os.path.join(OUTPUT_BASE, "train.log"), console=True, file=True
)

if __name__ == "__main__":
    Trainer(
        ALL_EXPS, 
        WandbRunMatcher(ENTITY, TRAINING_PROJECT, MISSING_VALUES), 
        parallel=PARALLEL, 
        local_artifact_dir="/share/j_sun/jth264/checkpoints"
    ).run()