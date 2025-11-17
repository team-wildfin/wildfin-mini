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
from fish_benchmark.execution.feature_extractor import FeatureExtractor
from fish_benchmark.execution.validator import Validator

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

if __name__ == "__main__":
    datasets = [CORALCAM, FISHFOLLOW]
    feature_extractor = FeatureExtractor(
        datasets=datasets,
        sliding_styles=SLIDING_STYLES,
        parallel=PARALLEL,
        check_report=CHECK_REPORT,
        device=device,
        logger=logger
    )
    feature_extractor.run(
        models=TARGET_MODELS,
        precomputed=PRECOMPUTED,
        validator = Validator(datasets=datasets, sliding_styles=SLIDING_STYLES).set_root(REPORT_ROOT)
    )