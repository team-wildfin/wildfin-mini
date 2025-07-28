import os
import subprocess
import yaml
import os 
from configs import VIDEO_MAE_WEIGHTED_EXPS, DINO_WEIGHTED_EXPS, RESNET50_WEIGHTED_EXPS
from typing import List, Optional, Callable
from fish_benchmark.types import TrainConfig
from fish_benchmark.utils import setup_logger
from submission import get_slurm_submission_command

PARALLEL = False
OUTPUT_BASE = os.path.join("logs", "train")
CONFIG_OUT = "generated_configs"
os.makedirs(CONFIG_OUT, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
logger = setup_logger("train", os.path.join(OUTPUT_BASE, "train.log"), console=True, file=True)

def save_config_to_file(config: TrainConfig, out_dir: str) -> str:
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
            gpu_count=1
        )
    else:
        command = f"python training/main.py --config {config_path}"

    logger.info(f"Running training with config: {config_id}\nCommand: {command}")
    subprocess.run(command, shell=True, check=True)

def dataset_filter(config: TrainConfig) -> bool:
    """
    Filter function to select configurations based on dataset.
    """
    return config.dataset in ['fishfollow']

def run(configs: List[TrainConfig], filter: Optional[Callable[[TrainConfig], bool]] = None):
    """
    Run training for a list of configurations, optionally filtering them.
    """
    for config in configs:
        if filter and not filter(config):
            continue
        config_path = save_config_to_file(config, CONFIG_OUT)
        run_training(config_path, config.id)


def main():
    """
    Main function to run training for predefined configurations.
    """
    logger.info("Starting training runs...")

    run(VIDEO_MAE_WEIGHTED_EXPS, filter=dataset_filter)

    run(DINO_WEIGHTED_EXPS, filter=dataset_filter)

    run(RESNET50_WEIGHTED_EXPS, filter=dataset_filter)

    logger.info("Training runs completed.")

if __name__ == "__main__":
    main()