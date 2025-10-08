"""
extracts features from precomputed inputs
"""

import yaml
import os
import torch
from config.models.models import MODEL_SLIDING_STYLES
from fish_benchmark.utils.general import setup_logger
import subprocess
import argparse
import shutil
from typing import Dict
from config.datasets import CORALCAM, FISHFOLLOW
from submission import get_slurm_submission_command

TARGET_MODELS = [
    "videomae",
    # 'dino',
    # 'dino_large',
    # 'resnet50'
]
SLIDING_STYLES = [
    # "frames",
    # "frames_w_temp",
    # "sliding_window",
    # "sliding_window_w_temp",
    # "sliding_window_w_stride",
    "sliding_window_ti8",
    # "fix_patched_512",
    # "test_frames",
    # "test_sliding_window",
    # "test_fix_patched_512",
    # "test_sliding_window_ti8"
]

PRECOMPUTED = False
PARALLEL = False
CHECK_REPORT = False
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
)


def get_wrap_command(
    source, dataset, sliding_style, dest_path, video_id, model, precomputed
):
    """
    SOURCE = args.source
    DATASET = args.dataset
    SLIDING_STYLE = args.sliding_style
    DEST_PATH = args.dest_path
    VIDEO_ID = args.id
    MODEL = args.model
    PRECOMPUTED = args.precomputed
    """
    return (
        f"python data/action/extract_features.py "
        f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
        f"--dataset {dataset} --model {model} --precomputed {precomputed} "
    )


def find_report(dataset, split, sliding_style, model):
    # report should exist in REPORT_ROOT/dataset/split/sliding_style/<model>_report.yml
    report_path = os.path.join(
        REPORT_ROOT, dataset, split, sliding_style, f"{model}_report.yml"
    )
    if not os.path.exists(report_path):
        logger.warning(
            f"Report not found for {dataset} {split} {sliding_style} {model} at {report_path}"
        )
        return None
    with open(report_path, "r") as f:
        report = yaml.safe_load(f)
    return report


def get_incomplete_subsets(report: Dict):
    # assert report is not empty
    assert report, "Report is empty"
    data: Dict = list(report.values())[
        0
    ]  # Assuming report is a dict with one key-value pair
    res = []
    for subset, subset_data in data.items():
        if subset_data["actual_files"] != subset_data["expected_items"]:
            res.append(subset)
    return res


def run(
    subset_path,
    subset_dest_path,
    ds_name: str,
    ss_name: str,
    split_name: str,
    subset_id: str,
    model: str,
    precomputed: bool,
):
    SUBSET_SOURCE = os.path.join(subset_path, "inputs") if precomputed else subset_path
    FEATURE_DEST = os.path.join(subset_dest_path, f"{model: str}_features")
    wrap_cmp = get_wrap_command(
        SUBSET_SOURCE, ds_name, ss_name, FEATURE_DEST, subset_id, model, precomputed
    )
    output_dir = os.path.join(OUT_ROOT, ds_name, ss_name, split_name, subset_id, model)
    submission_name = f"{ds_name}_{ss_name}_{split_name}_{subset_id}_{model}"
    command = (
        get_slurm_submission_command(submission_name, output_dir, wrap_cmp, gpu_count=1)
        if PARALLEL
        else wrap_cmp
    )
    logger.info(f"Running command for {submission_name} with command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command for {submission_name}: {e}")


def main():
    for dataset in [CORALCAM, FISHFOLLOW]:
        for split in dataset.splits:
            for model in TARGET_MODELS:
                for ss_name in set(split.get_sliding_style_names()) & set(MODEL_SLIDING_STYLES[model]):
                    SOURCE_PATH = (
                        os.path.join(dataset.precomputed_path, ss_name, split)
                        if PRECOMPUTED
                        else os.path.join(dataset.path, split)
                    )

                    DEST_PATH = os.path.join(dataset.precomputed_path, ss_name, split)

                    if CHECK_REPORT:
                        report = find_report(dataset.name, split.name, ss_name, model)
                        filt = get_incomplete_subsets(report) if report else {}
                        logger.info(
                            f"found {len(filt)} incomplete subsets for {dataset.name} {split.name} {ss_name} {model}: {filt}"
                        )
                    else:
                        filt = None

                    for subset in os.listdir(SOURCE_PATH):
                        if filt and subset not in filt:
                            logger.debug(
                                f"Skipping {subset} for {dataset.name} {split.name} {ss_name} {model}"
                            )
                            continue
                        subset_path = os.path.join(SOURCE_PATH, subset)
                        SUBSET_DEST_PATH = os.path.join(DEST_PATH, subset)
                        run(
                            subset_path,
                            SUBSET_DEST_PATH,
                            dataset.name,
                            ss_name,
                            split.name,
                            subset,
                            model,
                            PRECOMPUTED,
                        )


if __name__ == "__main__":
    main()
