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
from functools import partial
from typing import Callable, Dict, List, Union
from dataclasses import dataclass
from runner import WandbRunner
from configs import VIDEOMAE_BALANCED_FOCAL
import logging
from torchmetrics.functional.classification import (
    multilabel_precision,
    multilabel_recall,
    multilabel_f1_score,
    multilabel_average_precision
)
from functools import reduce
from fish_benchmark.utils import setup_logger
logger = setup_logger("export", "logs/export.log", console=True, file=True, level=logging.INFO)

# ==== CONFIG ====
ENTITY = "fish-benchmark"
DATASET = 'coralcam'
PROJECT = f"{DATASET}_eval"
LABEL_TOLERANCES = [0, 1, 3, 5, 7]
PARALLEL = False
DOWNLOAD_DIR = "test_metrics"
OUTPUT_PATH = 'results'

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
@dataclass
class Metric:
    name: str
    fn: Callable  # should accept (preds, targets, **kwargs)
    kwargs: Dict  # fixed args like average='micro'

class MetricCalculator: 
    def __init__(self, probs: torch.Tensor, targets: torch.Tensor):
        '''
        probs and targets should have the same shape of (n, d), where n is the number of samples and d is the number of classes 
        '''
        self.probs = probs
        self.targets = targets

    def flood_1d(self, bits, dis):
        res = np.zeros_like(bits)
        last = None
        for i in range(len(bits)):
            if bits[i] == 1:
                last = i
                res[i] = 1
            elif last is not None and i - last <= dis:
                res[i] = 1
        return res

    def flood(self, bits, dis):
        left = self.flood_1d(bits, dis)
        right = self.flood_1d(bits[::-1], dis)[::-1]
        return np.logical_or(left, right)

    def flood_all_columns(self, targets, dis):
        # targets: (n, d), apply flood along axis 0 per column
        flood_func = partial(self.flood, dis=dis)
        return np.apply_along_axis(flood_func, axis=0, arr=targets)
    
    def compute(self, metrics: List[Metric], label_tolerance = 0, column_subset = None, prefix = None) -> Dict[str, Union[float, List[float]]]:
        results = {}
        probs = self.probs if column_subset is None else self.probs[:, column_subset]
        targets = self.targets if column_subset is None else self.targets[:, column_subset]
        #logger.info(f"flooding with label tolerance {label_tolerance} on {targets.shape[0]} samples and {targets.shape[1]} classes")
        transformed_targets = torch.from_numpy(self.flood_all_columns(targets.cpu().numpy(), label_tolerance)) 
        assert probs.shape == transformed_targets.shape, f"probs shape {probs.shape} and targets shape {transformed_targets.shape} should match"
        num_classes = probs.shape[1]
        #logger.info(f"calculating {len(metrics)} metrics")
        for metric in metrics: 
            if "num_labels" in metric.kwargs.keys(): metric.kwargs["num_labels"] = num_classes
            output = metric.fn(probs, transformed_targets, **metric.kwargs)
            assert isinstance(output, torch.Tensor), f"Output of {metric.name} should be a tensor, but got {type(output)}"
            result_name = f'{prefix}_{metric.name}' if prefix else metric.name
            results[result_name] = output.tolist() if output.ndim > 0 else output.item()            
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

def compute_with_label_tolerance(results, label_tolerance, output_path):
    api = wandb.Api()
    rows = []
    for run_id, data in results.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        probs = torch.tensor(data["probs"])
        targets = torch.tensor(data["targets"])
        calc = MetricCalculator(
            probs=probs,
            targets=targets
        )
        num_classes = probs.shape[1]
        metrics = [
            #num_labels would be dynamically determined by the shape of probs
            Metric("f1_micro", multilabel_f1_score, {"num_labels": None, "average": "micro"}),
            Metric("f1_macro", multilabel_f1_score, {"num_labels": None, "average": "macro"}),
            Metric("precision_micro", multilabel_precision, {"num_labels": None, "average": "micro"}),
            Metric("precision_macro", multilabel_precision, {"num_labels": None, "average": "macro"}),
            Metric("recall_micro", multilabel_recall, {"num_labels": None, "average": "micro"}),
            Metric("recall_macro", multilabel_recall, {"num_labels": None, "average": "macro"}),
            Metric("mAP", multilabel_average_precision, {"num_labels": None, "average": "macro"}),
            Metric("acc", lambda x,y: ((x > 0.5) == y).float().mean(), {}),
            Metric("f1_per_class", multilabel_f1_score, {"num_labels": None, "average": None}),
            Metric("mAP_per_class", multilabel_average_precision, {"num_labels": None, "average": None}),
            Metric("precision_per_class", multilabel_precision, {"num_labels": None, "average": None}),
            Metric("recall_per_class", multilabel_recall, {"num_labels": None, "average": None}),
            Metric("positive_per_class", lambda _,y: (y.sum(dim=0)).float(), {}),
        ]
        
        aggregate_results = calc.compute(metrics, label_tolerance=label_tolerance)
        per_group_results = reduce(lambda a, b: a | b, 
                                   [calc.compute(metrics, 
                                                 label_tolerance=label_tolerance, 
                                                 column_subset=subgroup_mappings[DATASET][k], 
                                                 prefix=k) 
                                                 for k in subgroup_mappings[DATASET].keys()]) 
        row = run.config | {"run_id": run.id} | aggregate_results | per_group_results
        rows.append(row)
            
    #write to csv
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Wrote {len(rows)} rows to {output_path}")

def main():
    runner = WandbRunner(ENTITY, PROJECT, None)
    matched_run_ids = runner.get_matched_run_ids(VIDEOMAE_BALANCED_FOCAL, non_duplicate=False)
    results = get_results(ENTITY, PROJECT, matched_run_ids.values())
    for label_tolerance in LABEL_TOLERANCES:
        output_path = os.path.join(OUTPUT_PATH, f"{DATASET}_results_label_tolerance_{label_tolerance}.csv")
        logger.info(f"Computing results with label tolerance {label_tolerance} and writing to {output_path}")
        compute_with_label_tolerance(results, label_tolerance, output_path)

if __name__ == "__main__":
    main()