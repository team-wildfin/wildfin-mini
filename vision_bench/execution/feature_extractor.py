"""
Class that extracts features from video frames using a specified model, 
storing them in a specific location. This is used to precompute features for frozen backbones. 
"""
"""
extracts features from precomputed inputs
"""

import yaml
import os
import torch
from vision_bench.utils.general import setup_logger
from config.maps.model_sliding_style import MODEL_SLIDING_STYLES
import subprocess
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.execution.shard_executor import ShardExecutor
from vision_bench.execution.validator import Validator
from typing import List, Optional 
from vision_bench.typing.types import LocalDataset
import logging
from config.logging.loggers import get_console_logger

class FeatureExtractor(ShardExecutor):
    def __init__(self, 
                 datasets: List[LocalDataset], 
                 sliding_styles: List[str], 
                 models: List[str], 
                 use_precomputed: bool = False,
                 validator: Optional[Validator] = None,
                 parallel: bool = False, 
                 logger: logging.Logger = get_console_logger()):
        """
        Args:
            datasets: List of datasets to extract features from
            sliding_styles: List of sliding styles to extract features for
            models: List of models to extract features with
            use_precomputed: Whether to use preprocessed(resized) inputs if available
            validator: Optional Validator to check for completed subsets before extraction
            parallel: Whether to run in parallel using SLURM
            logger: Logger for logging messages

        """
        super().__init__(
            datasets=datasets,
            sliding_styles=sliding_styles,
            parallel=parallel,
            logger=logger,
        )
        self.models = models
        self.use_precomputed = use_precomputed
        self.validator = validator

    def set_default_validator(self, validator_root: str): 
        self.validator = Validator(
            datasets=self.datasets,
            sliding_styles=self.sliding_styles,
            root_path=validator_root,
            logger=self.logger
        )
        return self

    def get_wrap_cmd(self, source, dataset, sliding_style, dest_path, video_id, model, precomputed):
        return (
            f"python vision_bench/scripts/extract_features.py "
            f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
            f"--dataset {dataset} --model {model} --precomputed {precomputed} "
        )

    def get_subset_data_source(self, dataset: LocalDataset, split_name: str, subset: str, sliding_style: str, precomputed):
        if precomputed: 
            try: 
                return self.ShardManager(dataset.precomputed_path).locate_shard(split_name, subset, sliding_style, 'inputs')
            except FileNotFoundError as e:
                self.logger.warning(f"Precomputed input not found for {dataset.name} {split_name} {subset} {sliding_style}: {e}\nfalling back to raw source")
        return self.ShardManager(dataset.path).subset_path(split_name, subset)
    
    def run(self): 
        for dataset in self.datasets: 
            raw_source = self.ShardManager(dataset.path)
            precomputed_source = self.ShardManager(dataset.precomputed_path)
            logging_source = self.ShardManager(os.path.join("logs", "extract_features"))
            for split in dataset.splits: 
                for model in self.models: 
                    for ss_name in set(split.get_sliding_style_names()) & set([ss.name for ss in MODEL_SLIDING_STYLES[model]]):
                        report = self.validator.find_report(dataset.name, split.name, ss_name, model) if self.validator else None
                        print(f"Report for {dataset.name} {split.name} {ss_name} {model}: {report}")
                        skip = Validator.get_complete_subsets(report) if self.validator else {}
                        for subset in raw_source.list_subsets(split.name):
                            if subset in skip:
                                self.logger.debug(
                                    f"Skipping {subset} for {dataset.name} {split.name} {ss_name} {model}"
                                )
                                continue
                            subset_path = self.get_subset_data_source(dataset, split.name, subset, ss_name, self.use_precomputed)
                            dest_path = precomputed_source.locate_shard(split=split.name, subset=subset, sliding_style=ss_name, shard_type=f"{model}_features")
                            os.makedirs(dest_path, exist_ok=True)
                            wrap_cmd = self.get_wrap_cmd(
                                subset_path, dataset.name, ss_name, dest_path, subset, model, self.use_precomputed
                            )
                            output_dir = logging_source.locate_shard(dataset.name, split.name, subset, ss_name)
                            submission_name = f"{dataset.name}_{ss_name}_{split.name}_{subset}_{model}"
                            command = (
                                get_slurm_submission_command(submission_name, output_dir, wrap_cmd, gpu_count=1)
                                if self.parallel
                                else wrap_cmd
                            )
                            self.logger.info(f"Running command for {submission_name} with command: {command}")
                            try:
                                subprocess.run(command, shell=True, check=True)
                            except subprocess.CalledProcessError as e:
                                self.logger.error(f"Error running command for {submission_name}: {e}")