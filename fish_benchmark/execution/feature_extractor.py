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
from fish_benchmark.utils.general import setup_logger
from config.maps.model_sliding_style import MODEL_SLIDING_STYLES
import subprocess
import argparse
import shutil
from typing import Dict
from config.data.datasets import CORALCAM, FISHFOLLOW
from fish_benchmark.utils.submission import get_slurm_submission_command
from fish_benchmark.execution.shard_executor import ShardExecutor
from fish_benchmark.execution.validator import Validator
from typing import List, Optional 

TARGET_MODELS = [
    # "videomae",
    # 'dino',
    # 'dino_large',
    'resnet50', 
    # 'dinov3_large',
    # "vjepa2"
]
SLIDING_STYLES = [
    # "frames",
    "frames_w_temp",
    # "sliding_window",
    # "sliding_window_w_temp",
    # "sliding_window_w_stride",
    # "sliding_window_ti8",
    # "fix_patched_512",
    "test_frames",
    # "test_sliding_window",
    # "test_fix_patched_512",
    # "test_sliding_window_ti8"
]

PRECOMPUTED = False
PARALLEL = True
CHECK_REPORT = True
device = "cuda" if torch.cuda.is_available() else "cpu"
REPORT_ROOT = os.path.join("data", "validation", "reports")
OUT_ROOT = os.path.join("logs", "extract_features")
if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT, exist_ok=True)
logger = setup_logger(
    "extract_features",
    os.path.join(OUT_ROOT, "extract_fishfollow_features.log"),
    console=True,
    file=False,
    level="DEBUG"
)

class FeatureExtractor(ShardExecutor):
    def get_wrap_cmd(self, source, dataset, sliding_style, dest_path, video_id, model, precomputed):
        return (
            f"python data/action/extract_features.py "
            f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
            f"--dataset {dataset} --model {model} --precomputed {precomputed} "
        )
    
    def run(self, 
            models: List[str], 
            validator: Optional[Validator] = None, 
            precomputed: bool = False): 
        for dataset in self.datasets: 
            raw_source = self.ShardManager(dataset.path)
            precomputed_source = self.ShardManager(dataset.precomputed_path)
            logging_source = self.ShardManager(os.path.join("logs", "extract_features"))
            for split in dataset.splits: 
                for model in models: 
                    for ss_name in set(split.get_sliding_style_names()) & set([ss.name for ss in MODEL_SLIDING_STYLES[model]]):
                        skip = validator.get_complete_subsets(
                                validator.find_report(dataset.name, split.name, ss_name, model)
                                ) if validator else {}
                        source = precomputed_source if precomputed else raw_source
                        for subset in source.list_subsets(split.name):
                            if subset in skip:
                                logger.debug(
                                    f"Skipping {subset} for {dataset.name} {split.name} {ss_name} {model}"
                                )
                                continue
                            subset_path = (source.locate_precomputed(split.name, subset, ss_name, 'inputs') 
                                           if precomputed else source.subset_path(split.name, subset))
                            dest_path = precomputed_source.locate_precomputed(split.name, subset, ss_name, f"{model}_features")
                            os.makedirs(dest_path, exist_ok=True)
                            wrap_cmd = self.get_wrap_cmd(
                                subset_path, dataset.name, ss_name, dest_path, subset, model, precomputed
                            )
                            output_dir = logging_source.locate_precomputed(dataset.name, split.name, subset, ss_name)
                            submission_name = f"{dataset.name}_{ss_name}_{split.name}_{subset}_{model}"
                            command = (
                                get_slurm_submission_command(submission_name, output_dir, wrap_cmd, gpu_count=1)
                                if PARALLEL
                                else wrap_cmd
                            )
                            logger.info(f"Running command for {submission_name} with command: {command}")
                            try:
                                subprocess.run(command, shell=True, check=True)
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Error running command for {submission_name}: {e}")