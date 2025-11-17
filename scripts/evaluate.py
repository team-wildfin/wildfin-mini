import os
import subprocess
import wandb
import logging
from fish_benchmark.management.matcher import WandbRunMatcher
from datetime import datetime, timezone
import pprint
from fish_benchmark.typing.experiment import Experiment
from config.experiments.cvpr import CVPR_EXPS
from fish_benchmark.utils.submission import get_slurm_submission_command
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.management.query import query_pending_evaluations

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
RERUN = False
ALL_EXPS = list(
    filter(
        lambda exp: (
            exp.dataset == TRAINING_PROJECT and 
            exp.backbone in ['vjepa2'] and 
            exp.pooling == 'attention'),
        CVPR_EXPS
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
    while(pending_eval := query_pending_evaluations(
        WandbRunMatcher(ENTITY, TRAINING_PROJECT), WandbRunMatcher(ENTITY, EVAL_PROJECT), ALL_EXPS, rerun = RERUN)):
        logger.info("Evaluating the first pending experiment...")
        exp_id, run_id = next(iter(pending_eval.items()))
        try: 
            eval(ENTITY, TRAINING_PROJECT, run_id)
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed for {exp_id}: {e}")

if __name__ == "__main__":
    main()
