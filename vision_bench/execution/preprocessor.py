"""
Class that handles processing of data: 
- verifying labels length and dimension matches the input data, and organizes the data into desired location. 
- Preprocess video frames into model input resolution. 
"""
import logging
import os
import subprocess
from typing import Optional, Type
from vision_bench.execution.validator import Validator
from vision_bench.management.manager import ShardManager, SourceManager
from vision_bench.typing.types import LocalDataset, SlidingStyle
from config.maps.sliding_style_test import TEST_NAME
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.execution.shard_executor import ShardExecutor
    
# Example config values (replace with loading from a file if needed)

class Preprocessor: 
    def __init__(self, 
                 source_manager: Type[SourceManager] = SourceManager,
                 shard_manager: Type[ShardManager] = ShardManager,
                 validator: Optional[Validator] = None,
                 logger: logging.Logger = None, 
                 parallel: bool = False): 
        self.source_manager = source_manager
        self.shard_manager = shard_manager
        self.logger = logger or logging.getLogger(__name__)
        self.parallel = parallel
        self.validator = validator
    
    @staticmethod
    def get_wrap_cmd(source, input_dest, label_dest, subset, dataset, sliding_style, save_input):
        return (
            f'python vision_bench/scripts/slide_window.py '
            f'--source "{source}" --input_dest "{input_dest}" --label_dest "{label_dest}" --id "{subset}" --dataset "{dataset}" '
            f'--save_input {save_input} --sliding_style "{sliding_style}"'
        )

    def run(self, dataset: LocalDataset, sliding_style: SlidingStyle, compute_for_test: bool = False, save_input: bool = False): 
        source_locator = self.source_manager(dataset.path)
        dest_locator = self.shard_manager(dataset.precomputed_path)
        logging_locator = self.shard_manager(os.path.join("logs", "slide_window"))
        for split in dataset.splits: 
            if not compute_for_test and split.name == "test":
                self.logger.info(f"Skipping test split for {dataset.name} since compute_for_test is False.")
                continue
            
            ss = (sliding_style if sliding_style in split.sliding_styles else 
                 (SLIDING_STYLES[TEST_NAME[sliding_style.name]]
                  if SLIDING_STYLES[TEST_NAME[sliding_style.name]] in split.sliding_styles 
                  else None))
            
            
            if save_input: 
                report = self.validator.find_report(dataset.name, split.name, ss.name, 'inputs') if self.validator else None
                if report is None: 
                    raise ValueError(f"No validation report found for {dataset.name} {split.name} {ss.name} {'inputs'}. Cannot determine which subsets to skip. Please run the validator first.")

            skip = Validator.get_complete_subsets(report) if report and self.validator else {}
            for subset in source_locator.list_subsets(split.name): 
                if subset in skip:
                    self.logger.debug(
                        f"Skipping {subset} for {dataset.name} {split.name} {ss.name} {'inputs'}"
                    )
                    continue
                source = source_locator.subset_path(split.name, subset)
                if not os.path.exists(source):
                    self.logger.error(f"Source path does not exist: {source}")
                    continue
                input_dest = dest_locator.locate_shard(sliding_style=ss.name, split=split.name, subset=subset, shard_type='inputs')
                label_dest = dest_locator.locate_shard(sliding_style=ss.name, split=split.name, subset=subset, shard_type='labels')
                output_dir = logging_locator.locate_shard(sliding_style=ss.name, split=split.name, subset=subset, shard_type='logs')
                os.makedirs(output_dir, exist_ok=True)
                wrap_cmd = Preprocessor.get_wrap_cmd(source, input_dest, label_dest, subset, dataset.name, ss.name, save_input)
                submission_name = f"{dataset.name}_{ss.name}_{split.name}_{subset}"
                command = get_slurm_submission_command(
                        submission_name, output_dir, wrap_cmd, gpu_count=0
                    ) if self.parallel else wrap_cmd
                self.logger.info(f"Running command for {dataset.name}_{ss.name}_{split.name}_{subset} with command: {command}")
                try: 
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error running command for {submission_name}: {e}")