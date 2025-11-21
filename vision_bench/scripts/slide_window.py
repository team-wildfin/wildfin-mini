import argparse
import os
import yaml
import torch
from vision_bench.data.builder import DatasetBuilder
from config.data.datasets import DATASETS
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.utils.general import frame_id_with_padding, setup_logger
from tqdm import tqdm
import numpy as np
import shutil
import csv
import logging
from vision_bench.data.preprocessors import TorchVisionPreprocessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input_dest", required=True)
    parser.add_argument("--label_dest", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sliding_style", required=True)
    parser.add_argument("--type", default="train")
    parser.add_argument("--save_input", default=False)
    return parser.parse_args()

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    args = get_args()
    SOURCE = args.source
    print("debug: ------")
    print(args.dataset)
    DATASET = DATASETS[args.dataset]
    INPUT_DEST = args.input_dest
    LABEL_DEST = args.label_dest
    SLIDING_STYLE = SLIDING_STYLES[args.sliding_style]
    SAVE_INPUT = True if args.save_input == 'True' else False
    ID = args.id
    
    # Delete old folders if they exist
    if SAVE_INPUT and os.path.exists(INPUT_DEST): shutil.rmtree(INPUT_DEST)
    if os.path.exists(LABEL_DEST): shutil.rmtree(LABEL_DEST)

    # Check if the path exists
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"The specified path does not exist: {SOURCE}")
    
    dataset = DatasetBuilder(
        path = SOURCE, 
        dataset = DATASET,
        sliding_style = SLIDING_STYLE, 
        precomputed = False, 
        input_transform = TorchVisionPreprocessor(), 
        only_labels = False if SAVE_INPUT else True
    ).build()
    
    #make destination folders
    if SAVE_INPUT: os.makedirs(INPUT_DEST, exist_ok=True)
    os.makedirs(LABEL_DEST, exist_ok=True)

    # Create a CSV file for labels
    tsv_path = os.path.join(LABEL_DEST, f"{ID}.tsv")
    tsv_file = open(tsv_path, "w", newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t')

    logger.info(f"Saving input to {INPUT_DEST}, label to {LABEL_DEST}")
    TOTAL = len(dataset)
    for i, (clip, label) in tqdm(enumerate(dataset)):
        label_np = label.clone().cpu().int().numpy()
        tsv_writer.writerow(label_np.tolist())
        if SAVE_INPUT: 
            clip_np = clip.clone().cpu().numpy()
            np.save(os.path.join(INPUT_DEST, f'{ID}_{frame_id_with_padding(i)}.npy'), clip_np)
        
    tsv_file.close()
    logger.info(f"Saved {TOTAL} clips to {INPUT_DEST} and labels to {LABEL_DEST}")