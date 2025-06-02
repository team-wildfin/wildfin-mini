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
from submission import get_slurm_submission_command

TARGET_MODELS = [
    'videomae', 
    'dino', 
    'dino_large'
]
TARGET_DATASETS = [
    # "abby", 
    "mike"
]
SLIDING_STYLES = [
    # "frames", 
    # "frames_w_temp", 
    # "sliding_window", 
    # "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "fix_patched_512", 
    # "test_frames", 
    "test_sliding_window", 
    "test_fix_patched_512",
]

PRECOMPUTED = False
PARALLEL = True

model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/dataset.yml", "r"))
sliding_style_config = yaml.safe_load(open("config/sliding_style.yml", "r"))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_ROOT = os.path.join('logs', 'extract_features')
logger = setup_logger(
    'extract_features', 
    os.path.join(OUT_ROOT, 'extract_features.log'), 
    console=False, 
    file=True
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
        f'python data/action_scripts/extract_features.py '
        f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
        f'--dataset {dataset} --model {model} --precomputed {precomputed} '
    )

def main():
    for DATASET in TARGET_DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for SPLIT in list(dataset_config[DATASET]['splits'].keys()):
                for MODEL in TARGET_MODELS:
                    if SLIDING_STYLE not in dataset_config[DATASET]['splits'][SPLIT]['sliding_styles']: continue
                    if SLIDING_STYLE not in model_config[MODEL]['sliding_styles']: continue

                    SOURCE_PATH = (os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, SPLIT) 
                            if PRECOMPUTED 
                            else os.path.join(dataset_config[DATASET]['path'], SPLIT))
                    
                    DEST_PATH = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, SPLIT)
                    for SUBSET in os.listdir(SOURCE_PATH):
                        SUBSET_PATH = os.path.join(SOURCE_PATH, SUBSET)
                        SUBSET_DEST_PATH = os.path.join(DEST_PATH, SUBSET)
                        SUBSET_SOURCE = os.path.join(SUBSET_PATH, 'inputs') if PRECOMPUTED else SUBSET_PATH
                        FEATURE_DEST = os.path.join(SUBSET_DEST_PATH, f'{MODEL}_features')
                        wrap_cmp = get_wrap_command(
                            SUBSET_SOURCE, DATASET, SLIDING_STYLE, FEATURE_DEST, SUBSET, MODEL, PRECOMPUTED
                        )
                        output_dir = os.path.join(OUT_ROOT, DATASET, SLIDING_STYLE, SPLIT, SUBSET, MODEL)
                        submission_name = f'{DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET}_{MODEL}'
                        command = (get_slurm_submission_command(submission_name, output_dir, wrap_cmp, gpu=1) 
                                   if PARALLEL 
                                   else wrap_cmp)   
                        logger.info(f"Running command for {submission_name} with command: {command}")
                        subprocess.run(command, shell=True, check=True)

if __name__ == '__main__':
    main()