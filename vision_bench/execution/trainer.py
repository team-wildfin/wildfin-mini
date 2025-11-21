"""
Trains a model on desired experimental configurations. 
"""

import os
import subprocess
import yaml
import os
from config.experiments.neurips import *
from typing import Optional
from vision_bench.typing.experiment import Experiment
from vision_bench.utils.general import setup_logger
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.management.query import query_pending_experiments
from vision_bench.execution.exp_executor import ExperimentExecutor

class Trainer(ExperimentExecutor): 
    @staticmethod
    def save_config_to_file(config: Experiment, out_dir: str) -> str:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{config.id}.yml")
        with open(out_path, "w") as f:
            yaml.dump(config.model_dump(), f, sort_keys=False)
        return out_path
    
    def run_script(self, matcher_id: str, config_path: str, config_id: str, slurm_output_base: Optional[str] = None):
        wrap_cmd = f"python vision_bench/scripts/train.py --matcher_id {matcher_id} --config {config_path}"
        if self.parallel:
            output_dir = os.path.join(slurm_output_base, config_id)
            submission_name = config_id
            command = get_slurm_submission_command(
                submission_name=submission_name,
                output_dir=output_dir,
                wrap_command=wrap_cmd,
                gpu_count=1,
            )
        else:
            command = wrap_cmd
        self.logger.info(f"Running training with config: {config_id}\nCommand: {command}")
        subprocess.run(command, shell=True, check=True)

    def execute(self, exp: Experiment, slurm_output_base: str, tmp_dir: str):
        self.logger.info(f"Running training for experiment: {exp.id}")
        config_path = Trainer.save_config_to_file(exp, tmp_dir)
        try: 
            self.run_script(self.train_matcher.id, config_path, exp.id, slurm_output_base)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed for {exp.id}: {e}")

    def run(self, 
            slurm_output_base: Optional[str] = None, 
            tmp_dir: str = "tmp_configs"): 
        if self.avoid_reruns: 
            while(pending_exps := query_pending_experiments(
                self.train_matcher, self.experiments)):
                exp = next(iter(pending_exps))
                self.execute(exp, slurm_output_base, tmp_dir)
        else:
            for exp in self.experiments:
                self.execute(exp, slurm_output_base, tmp_dir)