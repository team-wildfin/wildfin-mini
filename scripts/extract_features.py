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
from config.data.datasets import CORALCAM
from vision_bench.utils.submission import get_slurm_submission_command
from vision_bench.execution.feature_extractor import FeatureExtractor
from config.main import VALIDATION_REPORTS_DIR

TARGET_MODELS = [
    'dinov3_base',
]
SLIDING_STYLES = [
    "frames_w_temp",
    "test_frames"
]
device = "cuda" if torch.cuda.is_available() else "cpu"
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
        datasets=[CORALCAM],
        sliding_styles=SLIDING_STYLES,
        models = TARGET_MODELS,
        logger=logger
    ).set_default_validator(validator_root=VALIDATION_REPORTS_DIR).run()