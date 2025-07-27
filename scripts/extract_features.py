'''
extracts features from precomputed inputs
'''

import yaml
import os 
import torch
from fish_benchmark.utils import setup_logger
import subprocess
import argparse
import shutil
from typing import Dict
from submission import get_slurm_submission_command

TARGET_MODELS = [
    # 'videomae', 
    # 'dino', 
    # 'dino_large', 
    'resnet50'
]
TARGET_DATASETS = [
    # "coralcam", 
    "fishfollow"
]
SLIDING_STYLES = [
    "frames", 
    # "frames_w_temp", 
    # "sliding_window", 
    # "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "fix_patched_512", 
    "test_frames", 
    # "test_sliding_window", 
    # "test_fix_patched_512",
]

PRECOMPUTED = False
PARALLEL = False
CHECK_REPORT = True

model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))
sliding_style_config = yaml.safe_load(open("config/sliding_style.yml", "r"))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
REPORT_ROOT = os.path.join('data', 'validation', 'reports')
OUT_ROOT = os.path.join('logs', 'extract_features')
if not os.path.exists(OUT_ROOT): 
    os.makedirs(OUT_ROOT, exist_ok=True)
logger = setup_logger(
    'extract_features', 
    os.path.join(OUT_ROOT, 'extract_fishfollow_features.log'), 
    console=True, 
    file=False
)

def get_wrap_command(source, dataset, sliding_style, dest_path, video_id, model, precomputed):
    '''
    SOURCE = args.source
    DATASET = args.dataset
    SLIDING_STYLE = args.sliding_style
    DEST_PATH = args.dest_path
    VIDEO_ID = args.id
    MODEL = args.model
    PRECOMPUTED = args.precomputed
    '''
    return (
        f'python data/action/extract_features.py '
        f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
        f'--dataset {dataset} --model {model} --precomputed {precomputed} '
    )

def find_report(dataset, split, sliding_style, model): 
    #report should exist in REPORT_ROOT/dataset/split/sliding_style/<model>_report.yml
    report_path = os.path.join(REPORT_ROOT, dataset, split, sliding_style, f'{model}_report.yml')
    if not os.path.exists(report_path):
        logger.warning(f"Report not found for {dataset} {split} {sliding_style} {model} at {report_path}")
        return None
    with open(report_path, 'r') as f:
        report = yaml.safe_load(f)
    return report

def get_incomplete_subsets(report: Dict): 
    #assert report is not empty
    assert report, "Report is empty"
    data: Dict = list(report.values())[0]  # Assuming report is a dict with one key-value pair
    res = []
    for subset, subset_data in data.items(): 
        if subset_data['actual_files'] != subset_data['expected_items']:
            res.append(subset)
    return res

def run(SUBSET_PATH, SUBSET_DEST_PATH, DATASET, SLIDING_STYLE, SPLIT, SUBSET, MODEL, PRECOMPUTED): 
    SUBSET_SOURCE = os.path.join(SUBSET_PATH, 'inputs') if PRECOMPUTED else SUBSET_PATH
    FEATURE_DEST = os.path.join(SUBSET_DEST_PATH, f'{MODEL}_features')
    wrap_cmp = get_wrap_command(
        SUBSET_SOURCE, DATASET, SLIDING_STYLE, FEATURE_DEST, SUBSET, MODEL, PRECOMPUTED
    )
    output_dir = os.path.join(OUT_ROOT, DATASET, SLIDING_STYLE, SPLIT, SUBSET, MODEL)
    submission_name = f'{DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET}_{MODEL}'
    command = (get_slurm_submission_command(submission_name, output_dir, wrap_cmp, gpu_count=1) 
                if PARALLEL 
                else wrap_cmp)   
    logger.info(f"Running command for {submission_name} with command: {command}")
    try: 
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command for {submission_name}: {e}")

def main():
    for DATASET in TARGET_DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for MODEL in TARGET_MODELS:
                for SPLIT in list(dataset_config[DATASET]['splits'].keys()):
                    if SLIDING_STYLE not in dataset_config[DATASET]['splits'][SPLIT]['sliding_styles']: continue
                    if SLIDING_STYLE not in model_config[MODEL]['sliding_styles']: continue

                    SOURCE_PATH = (os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, SPLIT) 
                            if PRECOMPUTED 
                            else os.path.join(dataset_config[DATASET]['path'], SPLIT))
                    
                    DEST_PATH = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, SPLIT)
                    
                    if CHECK_REPORT: 
                        report = find_report(DATASET, SPLIT, SLIDING_STYLE, MODEL)
                        filt = get_incomplete_subsets(report) if report else {}
                        logger.info(f"found {len(filt)} incomplete subsets for {DATASET} {SPLIT} {SLIDING_STYLE} {MODEL}: {filt}")
                    else: 
                        filt = None

                    for SUBSET in os.listdir(SOURCE_PATH):
                        if filt and SUBSET not in filt: 
                            logger.debug(f"Skipping {SUBSET} for {DATASET} {SPLIT} {SLIDING_STYLE} {MODEL}")
                            continue
                        SUBSET_PATH = os.path.join(SOURCE_PATH, SUBSET)
                        SUBSET_DEST_PATH = os.path.join(DEST_PATH, SUBSET)
                        run(
                            SUBSET_PATH, SUBSET_DEST_PATH, DATASET, SLIDING_STYLE, SPLIT, SUBSET, MODEL, PRECOMPUTED
                        )
                    

if __name__ == '__main__':
    main()