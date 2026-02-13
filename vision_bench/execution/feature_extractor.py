"""
Class that extracts features from video frames using a specified model, 
storing them in a specific location. This is used to precompute features for frozen backbones. 
"""
"""
extracts features from precomputed inputs
"""

import os
from config.maps.sliding_style_test import TEST_NAME
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.management.manager import ShardManager, SourceManager
from vision_bench.utils.general import setup_logger
from config.maps.model_sliding_style import MODEL_SLIDING_STYLES
import subprocess
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.execution.shard_executor import ShardExecutor
from vision_bench.execution.validator import Validator
from typing import List, Optional, Type 
from vision_bench.typing.types import LocalDataset, SlidingStyle
import logging

class FeatureExtractor:
    def __init__(self, 
                 use_precomputed: bool = False,
                 shard_manager: Type[ShardManager] = ShardManager,
                 source_manager: Type[SourceManager] = SourceManager,
                 validator: Optional[Validator] = None,
                 parallel: bool = False, 
                 logger: logging.Logger = None):
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
        self.use_precomputed = use_precomputed
        self.validator = validator
        self.shard_manager = shard_manager
        self.source_manager = source_manager
        self.parallel = parallel
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f"validator is set to {self.validator.root_path}")


    @staticmethod
    def get_wrap_cmd(source, dataset, sliding_style, dest_path, video_id, model, precomputed):
        return (
            f"python vision_bench/scripts/extract_features.py "
            f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
            f"--dataset {dataset} --model {model} --precomputed {precomputed} "
        )

    def get_subset_data_source(self, dataset: LocalDataset, split_name: str, subset: str, sliding_style: str, precomputed):
        if precomputed: 
            try: 
                return self.shard_manager(dataset.precomputed_path).locate_shard(split_name, subset, sliding_style, 'inputs')
            except FileNotFoundError as e:
                self.logger.warning(f"Precomputed input not found for {dataset.name} {split_name} {subset} {sliding_style}: {e}\nfalling back to raw source")
        return self.source_manager(dataset.path).subset_path(split_name, subset)
    
    def run(self, dataset: LocalDataset, sliding_style: SlidingStyle, model: str): 
        '''
        sliding_style: name of the training sliding style. Testing sliding style is inferred from this.
        '''
        raw_source = self.source_manager(dataset.path)
        precomputed_source = self.shard_manager(dataset.precomputed_path)
        logging_source = self.shard_manager(os.path.join("logs", "extract_features"))
        self.logger.info(f"Validator is set to: {self.validator.root_path}")
        for split in dataset.splits: 
            ss = (sliding_style if sliding_style in split.sliding_styles else 
                 (SLIDING_STYLES[TEST_NAME[sliding_style.name]]
                  if SLIDING_STYLES[TEST_NAME[sliding_style.name]] in split.sliding_styles 
                  else None))
            if not ss or ss.name not in [s.name for s in MODEL_SLIDING_STYLES[model]]: 
                self.logger.warning(f"No valid sliding style found for {dataset.name} {split.name} with requested style {sliding_style.name}. Skipping this split.")
                continue
            report = self.validator.find_report(dataset.name, split.name, ss.name, model) if self.validator else None
            if report is None: 
                raise ValueError(f"No validation report found for {dataset.name} {split.name} {ss.name} {model}. Cannot determine which subsets to skip. Please run the validator first.")
            self.logger.debug(f"Report for {dataset.name} {split.name} {ss.name} {model}: {report}")
            skip = Validator.get_complete_subsets(report) if self.validator else {}
            for subset in raw_source.list_subsets(split.name):
                if subset in skip:
                    self.logger.debug(
                        f"Skipping {subset} for {dataset.name} {split.name} {ss.name} {model}"
                    )
                    continue
                subset_path = self.get_subset_data_source(dataset, split.name, subset, ss.name, self.use_precomputed)
                dest_path = precomputed_source.locate_shard(split=split.name, subset=subset, sliding_style=ss.name, shard_type=f"{model}_features")
                os.makedirs(dest_path, exist_ok=True)
                wrap_cmd = self.get_wrap_cmd(
                    subset_path, dataset.name, ss.name, dest_path, subset, model, self.use_precomputed
                )
                output_dir = logging_source.locate_shard(dataset.name, split.name, subset, ss.name)
                submission_name = f"{dataset.name}_{ss.name}_{split.name}_{subset}_{model}"
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