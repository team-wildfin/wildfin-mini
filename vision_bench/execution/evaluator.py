"""
Evaluates model based on specified configuration if finished training runs are avaiable. 
Produces model predictions on the test set, along with the correct labels, and stores these vectors
for later use in metric computation. 
"""
import os
import subprocess
import logging
from vision_bench.management.wandb_matcher import WandbRunMatcher
from vision_bench.typing.experiment import Experiment
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.utils.general import setup_logger
from vision_bench.management.query import query_pending_evaluations, query_trained
from vision_bench.execution.exp_executor import ExperimentExecutor
from typing import List, Optional 

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

class Evaluator(ExperimentExecutor): 
    def get_wrap_cmd(self, run_id):
        return (
            f'python vision_bench/scripts/evaluate.py '
            f'--train_matcher_id {self.train_matcher.id} --eval_matcher_id {self.eval_matcher.id} --run {run_id}'
        )

    def execute(self, run_id: str):
        try:
            wrap_cmd = self.get_wrap_cmd(run_id)
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
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed for run {run_id}: {e}")

    def run(self): 
        if self.avoid_reruns: 
            while(pending_eval := query_pending_evaluations(
                self.train_matcher, self.eval_matcher, self.experiments)):
                self.logger.info("Evaluating the first pending experiment...")
                run_id = next(iter(pending_eval.items()))[1] # pending eval returns {exp_id: train_run_id}, we take the first train_run_id to evaluate
                self.execute(run_id)
        else: 
            for exp in self.experiments: 
                trained = query_trained(self.train_matcher, [exp])
                if len(trained[exp.id]) == 0:
                    self.logger.info(f"No trained runs found for experiment {exp.id}, skipping evaluation.")
                    continue
                run_id = self.train_matcher.get_latest(trained[exp.id])
                self.execute(run_id)

