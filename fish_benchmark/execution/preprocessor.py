"""
Class that handles processing of data: 
- verifying labels length and dimension matches the input data, and organizes the data into desired location. 
- Preprocess video frames into model input resolution. 
"""
import os
import subprocess
import yaml
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.utils.submission import get_slurm_submission_command
from config.data.datasets import CORALCAM, FISHFOLLOW
from fish_benchmark.data.shard_manager import NestedFSShardManager
from typing import List, Type
import logging
from fish_benchmark.data.shard_manager import ShardManager
from fish_benchmark.typing.types import LocalDataset
from fish_benchmark.execution.shard_executor import ShardExecutor
# Example config values (replace with loading from a file if needed)

class Preprocessor(ShardExecutor): 
    def get_wrap_cmd(self, source, input_dest, label_dest, subset, dataset, sliding_style, save_input):
        return (
            f'python data/action/slide_window.py '
            f'--source "{source}" --input_dest "{input_dest}" --label_dest "{label_dest}" --id "{subset}" --dataset "{dataset}" '
            f'--save_input {save_input} --sliding_style "{sliding_style}"'
        )

    def run(self, save_input: bool = False): 
        for dataset in self.datasets:
            source_locator = self.ShardManager(dataset.path)
            dest_locator = self.ShardManager(dataset.precomputed_path)
            logging_locator = self.ShardManager(os.path.join("logs", "slide_window"))
            for split in dataset.splits: 
                for ss_name in set(self.sliding_styles).intersection(set(split.get_sliding_style_names())):
                    for subset in source_locator.list_subsets(split.name): 
                        source = source_locator.subset_path(split.name, subset)
                        if not os.path.exists(source):
                            self.logger.error(f"Source path does not exist: {source}")
                            continue
                        input_dest = dest_locator.locate_precomputed(dataset.name, split.name, subset, ss_name, shard_type='inputs')
                        label_dest = dest_locator.locate_precomputed(dataset.name, split.name, subset, ss_name, shard_type='labels')
                        output_dir = logging_locator.locate_precomputed(dataset.name, split.name, subset, ss_name)
                        os.makedirs(output_dir, exist_ok=True)
                        wrap_cmd = self.get_wrap_cmd(source, input_dest, label_dest, subset, dataset.name, ss_name, save_input)
                        submission_name = f"{dataset.name}_{ss_name}_{split.name}_{subset}"
                        command = get_slurm_submission_command(
                                submission_name, output_dir, wrap_cmd, gpu_count=0
                            ) if self.parallel else wrap_cmd
                        self.logger.info(f"Running command for {dataset.name}_{ss_name}_{split.name}_{subset} with command: {command}")
                        try: 
                            subprocess.run(command, shell=True, check=True)
                        except subprocess.CalledProcessError as e:
                            self.logger.error(f"Error running command for {submission_name}: {e}")