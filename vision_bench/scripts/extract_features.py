from vision_bench.model.models import ModelBuilder
from config.maps.backbone_preprocessors import PREPROCESSORS
from vision_bench.data.builder import DatasetBuilder
import yaml 
import argparse
from torch.utils.data import DataLoader
import torch
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from vision_bench.utils.general import frame_id_with_padding
from config.data.datasets import DATASETS 
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.utils.debug import step_timer
import numpy as np
import shutil

BATCH_SIZE = 32
PROFILE = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#This script is runned like this: 
# 'python data/action_scripts/extract_features.py --subset_path "{SUBSET_PATH}" --label_path "{LABEL_PATH}" --dest_path "{DEST_PATH}" --id "{SUBSET_ID}" '
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest_path", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--sliding_style", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precomputed", default=False)
    #dataset is inferred from the subset_path
    return parser.parse_args()

def save_feature(dest, save_name, feature_tensor):
    path = os.path.join(dest, f"{save_name}.npy")
    np.save(path, feature_tensor.cpu().numpy())

def parallel_save_features(outputs, dest_path, video_id, start_frame_id):
    futures = []
    os.makedirs(dest_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for j in range(outputs.shape[0]):
            feature_tensor = outputs[j]
            #print(feature_tensor.shape)
            save_name = f"{video_id}_{frame_id_with_padding(start_frame_id + j)}"
            f = executor.submit(save_feature, dest_path, save_name, feature_tensor)
            futures.append(f)

        # Wait for all jobs to complete
        for f in futures:
            f.result()

if __name__ == '__main__':
    args = get_args()
    SOURCE = args.source
    DATASET = DATASETS[args.dataset]
    SLIDING_STYLE = SLIDING_STYLES[args.sliding_style]
    DEST_PATH = args.dest_path
    VIDEO_ID = args.id
    MODEL = args.model
    PRECOMPUTED = True if args.precomputed == 'True' else False
    if os.path.exists(DEST_PATH): shutil.rmtree(DEST_PATH)
    
    model = ModelBuilder.build(
        backbone_name=MODEL,
        pooler_name='mean',
        freeze_backbone=True
    ).to(device)
    input_transform = PREPROCESSORS[MODEL]

    dataset = DatasetBuilder(
        path=SOURCE,
        dataset=DATASET,
        sliding_style=SLIDING_STYLE, 
        input_transform=input_transform,
        precomputed=PRECOMPUTED
    ).build()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    print("loaded dataloader")
    for i, (batch_clip, _) in tqdm(enumerate(dataloader)):
        #print(f"Processing batch {i + 1}/{len(dataloader)}")
        batch_clip = batch_clip.to(device)
        with step_timer("Feature Extraction", verbose=PROFILE), torch.no_grad():
            outputs = model(batch_clip)
        with step_timer("Saving Features", verbose=PROFILE):
            parallel_save_features(outputs, DEST_PATH, VIDEO_ID, i * BATCH_SIZE)