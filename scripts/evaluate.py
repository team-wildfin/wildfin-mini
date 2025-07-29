import os
import subprocess
import wandb
import logging
from runner import WandbRunner
from datetime import datetime, timezone

from fish_benchmark.types import Experiment
from configs import (
    VIDEOMAE_WEIGHTED_EXPS,
    DINO_WEIGHTED_EXPS,
    RESNET50_WEIGHTED_EXPS,
    VIDEOMAE_BALANCED_FOCAL
)
from submission import get_slurm_submission_command
from fish_benchmark.utils import setup_logger

logger = setup_logger("evaluate", "logs/evaluate.log", console=True, file=True, level=logging.INFO)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
EVAL_PROJECT = f"{TRAINING_PROJECT}_eval"
PARALLEL = False

ALL_CONFIGS = VIDEOMAE_BALANCED_FOCAL

def get_wrap_cmd(entity, project, run_id):
    return (
        f'python evaluation/main.py '
        f'--entity {entity} --project {project} --run {run_id} '
    )

def eval(entity: str, project: str, run_id: str):
    wrap_cmd = get_wrap_cmd(entity, project, run_id)
    cmd = (
        get_slurm_submission_command(
            f"{run_id}",
            os.path.join("logs", "test", run_id),
            wrap_cmd,
            gpu_count=1,
        )
        if PARALLEL else wrap_cmd
    )
    logger.info(f"Running evaluation for {run_id} with command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    runner = WandbRunner(ENTITY, TRAINING_PROJECT, EVAL_PROJECT)
    matched_run_ids = runner.get_matched_run_ids(ALL_CONFIGS, non_duplicate=True)
    for run_id in matched_run_ids.values():
        logger.info(f"Running evaluation for run ID: {run_id}")
        eval(ENTITY, TRAINING_PROJECT, run_id)
