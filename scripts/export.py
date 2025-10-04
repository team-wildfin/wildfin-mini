#https://wandb.ai/fish-benchmark/coralcam_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os
from unittest import runner 
import yaml
import wandb
import csv
import json
import torch 
import numpy as np
from typing import Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from scripts.matcher import WandbRunMatcher
from configs import (
    VIDEOMAE_WEIGHTED_EXPS,
    DINO_WEIGHTED_EXPS, 
    RESNET50_WEIGHTED_EXPS,
)
import logging
from functools import reduce
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.utils.export import *
import yaml
from config.datasets import DATASETS
logger = setup_logger("export", "logs/export.log", console=True, file=True, level=logging.INFO)

# ==== CONFIG ====
ENTITY = "fish-benchmark"
DATASET_NAME = 'coralcam'
PROJECT = f"{DATASET_NAME}_eval"
dataset = DATASETS[DATASET_NAME]
LABEL_TOLERANCES = [0, 1, 3, 5, 7]
PARALLEL = False
DOWNLOAD_DIR = "test_metrics"
OUTPUT_PATH = 'results'
ALL_EXPS = (
    VIDEOMAE_WEIGHTED_EXPS +
    DINO_WEIGHTED_EXPS +
    RESNET50_WEIGHTED_EXPS
)

subgroup_mappings = {
    "coralcam": {
        'biting': [0, 1], 
        'aggression': [3, 4]
    }, 
    "fishfollow":{
        'habitat': [16, 17, 19], 
        'biting': [1, 2, 6, 9], 
        'movement': [3, 4], 
        'foraging': [8, 11, 14], 
        'interactions': [5, 7, 13], 
        'other': [0, 10, 12, 15, 18]
    }
}

from typing import Callable, Dict, List, Union
import torch
import numpy as np
from functools import partial

def compute(
    probs: torch.Tensor, 
    targets: torch.Tensor, 
    metrics: Dict[str, Metric],
    prefix: str = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, List]]:
    results = {}
    for name, metric_fn in metrics.items():
        key = name
        if prefix:
            key = f"{prefix}_{key}"
        value = metric_fn(probs, targets)
        results[key] = value.tolist() if isinstance(value, torch.Tensor) else value
    return results

    
def get_results(entity: str, project: str, run_ids: List[str]) -> Dict[str, dict]:
    results = {}
    api = wandb.Api()
    for run_id in run_ids:
        try:
            with open(f'logs/test_metrics/{run_id}.json', "r") as f:
                data = json.load(f)
                logger.info(f"Loaded locally {run_id}")
        except Exception as e:
            logger.info(f"Failed to find local file {run_id}: {e}")
            logger.info(f"Downloading artifact for {run_id}")
            artifact_name = f"test_metrics_{run_id}.json"
            artifact_path = f"{entity}/{project}/{artifact_name}:v0"
            artifact = api.artifact(artifact_path)
            file_path = artifact.download(root=DOWNLOAD_DIR)
            local_path = os.path.join(file_path, run_id + ".json")
            with open(local_path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded JSON for {run_id}")
        results[run_id] = data
    return results

def expand_confusion_matrices(matrices: torch.Tensor, names: List[str]) -> Dict[str, List[List[int]]]:
    """
    Expand confusion matrices to a dictionary with names as keys.
    Requires: tensor has shape [num_classes, 2, 2] where num_classes is the number of classes.
    """
    assert matrices.ndim == 3 and matrices.shape[1] == 2 and matrices.shape[2] == 2, \
        "Confusion matrix tensor must have shape [num_classes, 2, 2]"
    expanded = {}
    for i, name in enumerate(names):
        expanded[name] = matrices[i].cpu().numpy().tolist()
    return expanded

def compute_with_label_tolerance(results, label_tolerance, output_path):
    api = wandb.Api()
    rows = []
    for run_id, data in results.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        probs = torch.tensor(data["probs"])
        targets = torch.tensor(data["targets"])
        targets = torch.from_numpy(flood_all_columns(targets.cpu().numpy(), label_tolerance)).to(torch.device('cpu'))

        aggregate_metrics = {
            #average metrics
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "mAP": mAP,
            "acc": acc,
        }
        per_class_metrics = {
            #per-class metrics
            "f1_per_class": f1_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "mAP_per_class": mAP_per_class,
            "positive_per_class": positive_per_class,
        }
        results = compute(probs, targets, aggregate_metrics | per_class_metrics, device=torch.device('cpu'))
        per_group_results = union(
                            [compute(probs[:, subgroup_mappings[DATASET_NAME][k]], 
                                     targets[:, subgroup_mappings[DATASET_NAME][k]], 
                                     aggregate_metrics, 
                                     prefix=k) 
                                    for k in subgroup_mappings[DATASET_NAME].keys()]) 
        confusion_matrix = binary_confusion_matrix(probs, targets)
        per_col_confusion = expand_confusion_matrices(confusion_matrix, dataset.categories)
        per_habitat_confusion = union([
            expand_confusion_matrices(
                binary_confusion_matrix(
                    probs[mask := (targets[:, habitat_idx] == 1)], 
                    targets[mask]),
                [name + f"__{dataset.categories[habitat_idx]}_habitat" for name in dataset.categories]
            ) for habitat_idx in subgroup_mappings[DATASET_NAME]['habitat']
        ]) if 'habitat' in subgroup_mappings[DATASET_NAME] else {}


        row = run.config | {"run_id": run.id} | results | per_group_results | per_col_confusion | per_habitat_confusion
        rows.append(row)

    existing_rows = load_existing_csv(output_path)
    updated_rows = update(existing_rows, rows, key="run_id")
    fieldnames = get_all_fieldnames(updated_rows)
    fill_missing_fields(updated_rows, fieldnames)
    write_csv(output_path, updated_rows, fieldnames)

def main():
    runner = WandbRunMatcher(ENTITY, PROJECT)
    matched_run_ids = runner.match(ALL_EXPS)
    results = get_results(ENTITY, PROJECT, matched_run_ids.values())
    for label_tolerance in LABEL_TOLERANCES:
        output_path = os.path.join(OUTPUT_PATH, f"{DATASET_NAME}_results_label_tolerance_{label_tolerance}.csv")
        logger.info(f"Computing results with label tolerance {label_tolerance} and writing to {output_path}")
        compute_with_label_tolerance(results, label_tolerance, output_path)

if __name__ == "__main__":
    main()