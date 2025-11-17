'''
test runs pytorch dataloader iterating batches on custom dataset
'''
print("importing torch")
from torch.utils.data import DataLoader
import torch
print("importing fish_benchmark")
from fish_benchmark.data.dataset import DatasetBuilder
from fish_benchmark.data.sampler import MultiLabelBalancedSampler
from config.data.datasets import DATASETS
from config.data.sliding_styles import SLIDING_STYLES
from fish_benchmark.utils.general import setup_logger 
from fish_benchmark.models import get_input_transform
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
    STYLE = SLIDING_STYLES[args.style]
    MODEL = args.model  # nullable
    PRECOMPUTED = True if args.precomputed == 'True' else False
    SPLIT = 'train'

    config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
    PATH = os.path.join(config[DATASET]['precomputed_path'], STYLE, SPLIT) if PRECOMPUTED else os.path.join(config[DATASET]['path'], SPLIT)
    print("initializing builder")
    builder = DatasetBuilder(
        path=PATH,
        dataset=DATASET,
        sliding_style=STYLE,
        precomputed=PRECOMPUTED, 
        feature_model=MODEL,
        only_labels=False 
    )
    print("building dataset")
    dataset = builder.build()
    sampler = MultiLabelBalancedSampler(dataset)

    # === Wrap in DataLoader ===
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # === Initialize class count tracker ===
    num_classes = dataset.label_tensor.shape[1]
    class_counts = torch.zeros(num_classes, dtype=torch.int32)

    print(f"Sampling from dataset with {num_classes} classes")

    # === Iterate and accumulate label counts ===
    for frames, labels in tqdm(loader, desc="Iterating over balanced DataLoader"):
        class_counts += labels.sum(dim=0).int()

    # === Report class sample counts ===
    print("\n📊 Sample counts per class from DataLoader iteration:")
    for i in range(num_classes):
        print(f"Class {i:2d}: {class_counts[i].item()} samples")

    empty_classes = (class_counts == 0).nonzero(as_tuple=True)[0].tolist()
    if empty_classes:
        print(f"\n⚠️ Warning: Some classes were never sampled: {empty_classes}")
    else:
        print("\n✅ All classes were sampled at least once.")