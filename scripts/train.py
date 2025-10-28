import os
import subprocess
import yaml
import os
from config.experiments.neurips import *
from typing import List, Optional, Callable
from fish_benchmark.typing.experiment import Experiment
from scripts.matcher import WandbRunMatcher
from fish_benchmark.utils.general import setup_logger
from submission import get_slurm_submission_command
from config.experiments.cvpr import CVPR_EXPS
import pprint
from scripts.query import query_pending_experiments

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


def save_config_to_file(config: Experiment, out_dir: str) -> str:
    out_path = os.path.join(out_dir, f"{config.id}.yml")
    with open(out_path, "w") as f:
        yaml.dump(config.model_dump(), f, sort_keys=False)
    return out_path


def run_training(config_path: str, config_id: str):
    if PARALLEL:
        output_dir = os.path.join(OUTPUT_BASE, config_id)
        submission_name = config_id
        wrap_cmd = f"python training/main.py --config {config_path}"
        command = get_slurm_submission_command(
            submission_name=submission_name,
            output_dir=output_dir,
            wrap_command=wrap_cmd,
            gpu_count=1,
        )
    else:
        command = f"python training/main.py --config {config_path}"
    logger.info(f"Running training with config: {config_id}\nCommand: {command}")
    subprocess.run(command, shell=True, check=True)



def main():
    """
    Main function to run training for predefined configurations.
    """
    logger.info("Starting training runs...")
    ALL_EXPS = [exp for exp in CVPR_EXPS if filt(exp)]
    while(pending_exps := query_pending_experiments(
        WandbRunMatcher(ENTITY, TRAINING_PROJECT), ALL_EXPS)):
        exp = next(iter(pending_exps))
        logger.info(f"Running training for experiment: {exp.id}")
        config_path = save_config_to_file(exp, CONFIG_OUT)
        try: 
            run_training(config_path, exp.id)
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed for {exp.id}: {e}")

if __name__ == "__main__":
    main()
