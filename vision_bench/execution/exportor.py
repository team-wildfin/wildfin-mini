""" 
Calculates the desired metrics for experimental evaluations. 
"""
#https://wandb.ai/fish-benchmark/coralcam_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os
import json
import torch 
from typing import Callable, Dict, List, Optional, Union
from vision_bench.management.wandb_matcher import WandbRunMatcher
import logging
from functools import reduce
from vision_bench.utils.general import setup_logger
from vision_bench.utils.export import *
from vision_bench.management.query import query_trained, query_evaluated
from vision_bench.management.wandb_matcher import WandbRunMatcher
from vision_bench.execution.exp_executor import ExperimentExecutor
from vision_bench.typing.experiment import Experiment
from vision_bench.typing.metric import Metric

logger = setup_logger("export", "logs/export.log", console=True, file=True, level=logging.INFO)
from typing import Callable, Dict, List, Union
import torch

class Exportor(ExperimentExecutor):
    """
    Calculates all aggregate metrics for all subgroups. As well as per class metrics for all classes. Writes results to a CSV file.
    """
    def __init__(self, 
                 experiments: List[Experiment],
                 train_matcher: WandbRunMatcher,
                 eval_matcher: WandbRunMatcher,
                 logger: Optional[logging.Logger] = None,
                 parallel: bool = False, 
                 aggregate_metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], Union[float, List]]] = {},
                 per_class_metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], Union[float, List]]] = {},
                 subgroup_mappings: Dict[str, List[int]] = {},
                    output_base: str = ".",
                    output_name: str = "results.csv",
                 ):
        super().__init__(
            experiments=experiments,
            train_matcher=train_matcher,
            eval_matcher=eval_matcher,
            logger=logger,
            parallel=parallel
        )
        self.aggregate_metrics = aggregate_metrics
        self.per_class_metrics = per_class_metrics
        self.subgroup_mappings = subgroup_mappings
        self.output_base = output_base
        self.output_name = output_name

    def get_results(self, matcher: WandbRunMatcher, run_ids: List[str]) -> Dict[str, dict]:
        results = {}
        for run_id in run_ids:
            data_path = matcher.get_artifact(local_path=f"{run_id}.json",
                                        remote_path = f"test_metrics_{run_id}.json:v0")
            with open(data_path, "r") as f:
                data = json.load(f)
            results[run_id] = {}
            results[run_id]['data'] = data
            results[run_id]['config'] = matcher.get_run_config(run_id)
        return results
    
    @staticmethod
    def compute_metrics(
        probs: torch.Tensor, 
        targets: torch.Tensor, 
        metrics: Dict[str, Metric],
        prefix: str = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Union[float, List]]:
        results = {}
        for name, metric in metrics.items():
            key = name
            if prefix:
                key = f"{prefix}_{key}"
            value = metric(probs, targets)
            results[key] = value.tolist() if isinstance(value, torch.Tensor) else value
        return results

    def compute(self, results, output_path):
        rows = []
        for run_id, dic in results.items():
            config = dic['config']
            data = dic['data']
            probs = torch.tensor(data["probs"])
            targets = torch.tensor(data["targets"])
            assert probs.shape == targets.shape, f"Probs and targets must have the same shape, got {probs.shape} and {targets.shape}"
            assert probs is not None and targets is not None, f"Probs and targets must not be None for run {run_id}"
            interested_cols = sorted(reduce(lambda acc, x: acc + x, self.subgroup_mappings.values(), []))
            results = Exportor.compute_metrics(probs[:, interested_cols], targets[:, interested_cols], 
                            self.aggregate_metrics | self.per_class_metrics, device=torch.device('cpu'))
            per_group_results = union(
                                [Exportor.compute_metrics(probs[:, self.subgroup_mappings[k]], 
                                        targets[:, self.subgroup_mappings[k]], 
                                        self.aggregate_metrics, 
                                        prefix=k) 
                                        for k in self.subgroup_mappings.keys()]) 
            row = config | {"run_id": run_id} | results | per_group_results
            rows.append(row)
        existing_rows = load_existing_csv(output_path)
        updated_rows = update(existing_rows, rows, key="run_id")
        fieldnames = get_all_fieldnames(updated_rows)
        fill_missing_fields(updated_rows, fieldnames)
        write_csv(output_path, updated_rows, fieldnames)

    def run(self): 
        trained = query_trained(self.train_matcher, self.experiments)
        evaluated = query_evaluated(self.eval_matcher, trained)
        runs = [self.eval_matcher.get_latest(v) for v in evaluated.values() if len(v) > 0]
        results = self.get_results(self.eval_matcher, runs)
        output_path = os.path.join(self.output_base, self.output_name)
        self.compute(results, output_path)