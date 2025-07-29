import os
import subprocess
import wandb
import logging
from scripts.matcher import WandbRunMatcher
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
ALL_EXPS = list(
    filter(
        lambda exp: exp.dataset == TRAINING_PROJECT,
        VIDEOMAE_WEIGHTED_EXPS + DINO_WEIGHTED_EXPS + RESNET50_WEIGHTED_EXPS,
    )
)
EVAL_PROJECT = f"{TRAINING_PROJECT}_eval"
PARALLEL = False

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

def main():
    train_project_matcher = WandbRunMatcher(ENTITY, TRAINING_PROJECT)
    eval_project_matcher = WandbRunMatcher(ENTITY, EVAL_PROJECT)

    trained = train_project_matcher.match(ALL_EXPS)     # {exp_id: run_id}
    evaluated = eval_project_matcher.match(ALL_EXPS)   # {exp_id: run_id}
    logger.info(f"Total experiments: {len(ALL_EXPS)}")
    logger.info(f"Found {len(trained)} trained runs and {len(evaluated)} evaluated runs.")

    pending_eval = {exp_id: trained[exp_id] for exp_id in set(trained.keys()) - set(evaluated.keys())}
    
    if not pending_eval:
        logger.info("All experiments have been evaluated.")
        return

    logger.info(f"{len(pending_eval)} experiments pending evaluation:")
    for exp_id in sorted(pending_eval.keys()):
        logger.info(f"  - {exp_id}")

    confirm = input("Proceed with evaluation for these experiments? [y/n]: ").strip().lower()
    if confirm != "y":
        logger.info("Aborting evaluation.")
        return

    for exp_id, run_id in pending_eval.items():
        logger.info(f"Running evaluation for run ID: {run_id} (experiment: {exp_id})")
        eval(ENTITY, TRAINING_PROJECT, run_id)


if __name__ == "__main__":
    main()
