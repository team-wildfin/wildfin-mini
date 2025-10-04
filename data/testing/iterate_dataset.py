'''
Test runs custom dataset iteration
'''
print("importing torch")
from torch.utils.data import DataLoader
import torch
print("importing fish_benchmark")
from fish_benchmark.data.dataset import DatasetBuilder
from fish_benchmark.data.sampler import MultiLabelBalancedSampler
from fish_benchmark.utils.general import setup_logger 
from fish_benchmark.models import get_input_transform
from config.datasets import DATASETS
print("importing utilities")
import yaml
import argparse
from tqdm import tqdm
import os
print("imported")

logger = setup_logger(
    "iterate_dataset", 
    "",
    console=True,
    file=False,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--precomputed", default=False)
    parser.add_argument("--model", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    DATASET = DATASETS[args.dataset]
    STYLE = args.style
    MODEL = args.model  # nullable
    PRECOMPUTED = True if args.precomputed == 'True' else False
    SPLIT = 'train'

    PATH = os.path.join(DATASET.precomputed_path, STYLE, SPLIT) if PRECOMPUTED else os.path.join(DATASET.path, SPLIT)
    print("initializing builder")
    builder = DatasetBuilder(
        path=PATH,
        dataset_name=DATASET,
        style=STYLE,
        precomputed=PRECOMPUTED, 
        feature_model=MODEL,
        only_labels=False 
    )
    print("building dataset")
    dataset = builder.build()
    frame_0, label_0 = next(iter(dataset))
    print("frame_0 shape:", frame_0.shape)
    print("label_0 shape:", label_0.shape)

    for frame, label in tqdm(dataset):
        pass