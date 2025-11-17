"""
Evaluates model based on specified configuration if finished training runs are avaiable. 
Produces model predictions on the test set, along with the correct labels, and stores these vectors
for later use in metric computation. 
"""
import os
import subprocess
import wandb
import logging
from fish_benchmark.management.wandb_matcher import WandbRunMatcher
from datetime import datetime, timezone
import pprint
from fish_benchmark.typing.experiment import Experiment
from config.experiments.cvpr import CVPR_EXPS
from fish_benchmark.utils.submission import get_slurm_submission_command
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.management.query import query_pending_evaluations
from fish_benchmark.execution.exp_executor import ExperimentExecutor
from fish_benchmark.management.matcher import Matcher

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

class Evaluator(ExperimentExecutor): 
    def get_wrap_cmd(self, entity, project, run_id):
        return (
            f'python evaluation/main.py '
            f'--entity {entity} --project {project} --run {run_id} '
        )

    def eval(self, matcher: Matcher, run_id: str):
        wrap_cmd = self.get_wrap_cmd(entity, project, run_id)
        cmd = (
            get_slurm_submission_command(
                f"{run_id}",
                os.path.join("logs", "test", run_id),
                wrap_cmd,
                gpu_count=1,
            )
            if self.parallel else wrap_cmd
        )
        logger.info(f"Running evaluation for {run_id} with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    def run(self): 
        while(pending_eval := query_pending_evaluations(
            self.train_matcher, self.eval_matcher, self.experiments)):
            self.logger.info("Evaluating the first pending experiment...")
            exp_id, run_id = next(iter(pending_eval.items()))
            try: 
                self.eval(self.eval_matcher, run_id)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Evaluation failed for {exp_id}: {e}")