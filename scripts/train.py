import os
import subprocess
import yaml
import os
from configs import *
from typing import List, Optional, Callable
from fish_benchmark.types import Experiment
from scripts.matcher import WandbRunMatcher
from fish_benchmark.utils.general import setup_logger
from submission import get_slurm_submission_command

PARALLEL = False
OUTPUT_BASE = os.path.join("logs", "train")
ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
ALL_EXPS = list(
    filter(
        lambda exp: exp.dataset == TRAINING_PROJECT,
        VIDEOMAE_WEIGHTED_EXPS
    )
)
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
    runner = WandbRunMatcher(ENTITY, TRAINING_PROJECT)
    matched_run_ids = runner.match(ALL_EXPS)
    pending_exps = [exp for exp in ALL_EXPS if exp.id not in matched_run_ids]

    logger.info(f"Total of {len(ALL_EXPS)} experiments")
    logger.info(f"{len(matched_run_ids)} already matched runs found.")
    logger.info(f"{len(pending_exps)} experiments pending training:\n")

    for exp in pending_exps:
        print(f"  - {exp.id}")

    if not pending_exps:
        logger.info("No pending experiments to run.")
        return

    user_input = input("\nProceed with running these experiments? (y/n): ").strip().lower()
    if user_input != "y":
        logger.info("Aborted by user.")
        return

    for exp in pending_exps:
        logger.info(f"Running training for experiment: {exp.id}")
        config_path = save_config_to_file(exp, CONFIG_OUT)
        run_training(config_path, exp.id)

    logger.info("Training runs completed.")



if __name__ == "__main__":
    main()
