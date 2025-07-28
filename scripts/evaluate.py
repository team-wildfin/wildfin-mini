import os
import subprocess
import wandb
import logging
from datetime import datetime, timezone

from fish_benchmark.types import TrainConfig
from configs import (
    VIDEOMAE_WEIGHTED_EXPS,
    DINO_WEIGHTED_EXPS,
    RESNET50_WEIGHTED_EXPS,
)
from submission import get_slurm_submission_command
from fish_benchmark.utils import setup_logger

logger = setup_logger("evaluate", "logs/evaluate.log", console=True, file=True, level=logging.INFO)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
EVAL_PROJECT = "coralcam_eval"
PARALLEL = False

ALL_CONFIGS = VIDEOMAE_WEIGHTED_EXPS + DINO_WEIGHTED_EXPS + RESNET50_WEIGHTED_EXPS


def match_config(run_config: dict, reference: TrainConfig) -> bool:
    """Check if a W&B run config matches a TrainConfig, ignoring 'id' and defaulting missing fields."""
    def get(run_config, key):
        if key == 'fulltune':
            return run_config.get(key, False)
        elif key == 'label_type':
            return run_config.get(key, 'onehot')
        return run_config.get(key)

    run_config = {k.lower(): v for k, v in run_config.items()}
    ref_dict = reference.model_dump(mode="json", exclude={"id"})

    for k, v in ref_dict.items():
        run_val = get(run_config, k)
        if run_val != v:
            logger.debug(f"Config mismatch: {k} -> {run_val} != {v} in {reference.id}")
            return False
    return True


def get_wrap_cmd(entity, project, run_id):
    return (
        f'python evaluation/main.py '
        f'--entity {entity} --project {project} --run {run_id} '
    )


def main():
    api = wandb.Api()

    logger.info(f"Fetching completed training runs from {ENTITY}/{TRAINING_PROJECT}...")
    training_runs = api.runs(f"{ENTITY}/{TRAINING_PROJECT}", filters={"state": "finished"})
    logger.info(f"Found {len(training_runs)} training runs.")

    logger.info(f"Fetching completed evaluation runs from {ENTITY}/{EVAL_PROJECT}...")
    eval_runs = api.runs(f"{ENTITY}/{EVAL_PROJECT}", filters={"state": "finished"})
    logger.info(f"Found {len(eval_runs)} evaluation runs.")

    # Build config-to-run-id mapping for existing evals
    already_evaluated = set()
    for run in eval_runs:
        for cfg in ALL_CONFIGS:
            if match_config(run.config, cfg):
                already_evaluated.add(cfg.id)
                break

    logger.info(f"{len(already_evaluated)} configs already evaluated.")

    matched_runs = {}
    for run in training_runs:
        for cfg in ALL_CONFIGS:
            if cfg.id in already_evaluated:
                logger.debug(f"Skipping {cfg.id} — already evaluated.")
                continue
            if match_config(run.config, cfg):
                if cfg.id not in matched_runs or run.created_at > matched_runs[cfg.id].created_at:
                    matched_runs[cfg.id] = run
                break

    logger.info(f"Found {len(matched_runs)} training runs still needing evaluation.")

    for key, run in sorted(matched_runs.items()):
        logger.info(f"{key} -> {run.id} @ {run.created_at}")

    for run in matched_runs.values():
        wrap_cmd = get_wrap_cmd(ENTITY, TRAINING_PROJECT, run.id)
        cmd = (
            get_slurm_submission_command(
                f"{run.id}",
                os.path.join("logs", "test", run.id),
                wrap_cmd,
                gpu_count=1,
            )
            if PARALLEL else wrap_cmd
        )
        logger.info(f"Running evaluation for {run.id} with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
