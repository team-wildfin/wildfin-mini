import os
import subprocess
import wandb
from fish_benchmark.types import TrainConfig
from configs import VIDEOMAE_WEIGHTED_EXPS, DINO_WEIGHTED_EXPS, RESNET50_WEIGHTED_EXPS, RESNET_RANDOM_FOCAL
from submission import get_slurm_submission_command
from fish_benchmark.utils import setup_logger
import logging
from datetime import datetime, timezone

logger = setup_logger("evaluate", "logs/evaluate.log", console=True, file=True, level = logging.INFO)

ENTITY = "fish-benchmark"
PROJECT = "coralcam"
PARALLEL = False

ALL_CONFIGS = VIDEOMAE_WEIGHTED_EXPS + DINO_WEIGHTED_EXPS + RESNET50_WEIGHTED_EXPS

def match_config(run_config: dict, reference: TrainConfig) -> bool:
    """Check if a W&B run config matches a TrainConfig, ignoring 'id' and defaulting missing fields like 'fulltune'."""
    def get(run_config, key): 
        if key == 'fulltune':
            return run_config.get(key, False)
        elif key == 'label_type': 
            return run_config.get(key, 'onehot')
        return run_config.get(key, None)


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
    logger.info(f"Fetching runs for {ENTITY}/{PROJECT}...")
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"})
    logger.info(f"Found {len(runs)} runs.")
    matched_runs = {}
    for run in runs:
        if run.id == 'udu2h10c':  # Example run ID to skip
            logger.debug(f"This is the testing run")
        for cfg in ALL_CONFIGS:
            if match_config(run.config, cfg):
                key = cfg.id
                if key not in matched_runs or run.created_at > matched_runs[key].created_at:
                    matched_runs[key] = run
                break

    logger.info(f"Matched {len(matched_runs)} runs with the provided configurations.")

    for key, run in sorted(matched_runs.items()):
        logger.info(f"{key} -> {run.id} @ {run.created_at}")

    for run in matched_runs.values():
        wrap_cmd = get_wrap_cmd(ENTITY, PROJECT, run.id)
        cmd = (
            get_slurm_submission_command(
                f"{run.id}",
                os.path.join("logs", "test", run.id),
                wrap_cmd,
                gpu_count=1,
            )
            if PARALLEL
            else wrap_cmd
        )
        print(f"Running command for {run.id} with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()