"""
extracts features from precomputed inputs
"""

import yaml
import os
import torch
from vision_bench.utils.general import setup_logger
from config.maps.model_sliding_style import MODEL_SLIDING_STYLES
import subprocess
import argparse
import shutil
from typing import Dict
from config.data.datasets import CORALCAM, FISHFOLLOW
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.execution.feature_extractor import FeatureExtractor

DATASETS = [CORALCAM, FISHFOLLOW]
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

if __name__ == "__main__":
    FeatureExtractor(
        datasets=DATASETS,
        sliding_styles=SLIDING_STYLES,
        models = TARGET_MODELS,
        logger=logger
    ).set_default_validator(validator_root=REPORT_ROOT).run()