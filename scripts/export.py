#https://wandb.ai/fish-benchmark/coralcam_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os
from fish_benchmark.management.wandb_matcher import WandbRunMatcher
from config.experiments.cvpr import CVPR_EXPS
import logging
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.utils.export import *
from config.data.datasets import DATASETS
from fish_benchmark.execution.exportor import Exportor
from config.experiments.defaults import MISSING_VALUES
logger = setup_logger("export", "logs/export.log", console=True, file=True, level=logging.INFO)

# ==== CONFIG ====
ENTITY = "fish-benchmark"
DATASET_NAME = 'coralcam'
PROJECT = f"{DATASET_NAME}"
EVAL_PROJECT = f"{DATASET_NAME}_eval"
dataset = DATASETS[DATASET_NAME]
LABEL_TOLERANCES = [7]
PARALLEL = False
DOWNLOAD_DIR = "test_metrics"
OUTPUT_PATH = 'results/new_grouping'
os.makedirs(OUTPUT_PATH, exist_ok=True)
ALL_EXPS = CVPR_EXPS

subgroup_mappings = {
    "coralcam": {
        'biting': [0, 1], 
        'aggression': [3]
    }, 
    "fishfollow":{
        "habitat": [16, 17, 19],
        "bites": [6, 1, 9],
        "movement": [3, 8, 11, 14, 15],
        "social_interaction": [5],
        "not_visible": [13],
    }
}

if __name__ == "__main__":
    for tolerance in LABEL_TOLERANCES: 
        aggregate_metrics = {
            #average metrics
            "f1_micro": F1Micro(tolerance),
            "f1_macro": F1Macro(tolerance),
            "precision_micro": PrecisionMicro(tolerance),
            "precision_macro": PrecisionMacro(tolerance),
            "recall_micro": RecallMicro(tolerance),
            "recall_macro": RecallMacro(tolerance),
            "acc": Accuracy(tolerance),
            "mAP": mAP(),
        }
        per_class_metrics = {
            #per-class metrics
            "f1_per_class": F1PerClass(tolerance),
            "precision_per_class": PrecisionPerClass(tolerance),
            "recall_per_class": RecallPerClass(tolerance),
            "positive_per_class": PositivePerClass(),
            "ap_per_class": APPerClass(),
            "confusion_matrix": binary_confusion_matrix,
        }
        exportor = Exportor(
            experiments=ALL_EXPS,
            train_matcher=WandbRunMatcher(ENTITY, PROJECT, MISSING_VALUES),
            eval_matcher=WandbRunMatcher(ENTITY, EVAL_PROJECT, MISSING_VALUES), 
            aggregate_metrics=aggregate_metrics,
            per_class_metrics=per_class_metrics,
            subgroup_mappings=subgroup_mappings[DATASET_NAME], 
            output_base=OUTPUT_PATH,
            output_name=f"{DATASET_NAME}_results_label_tolerance_{tolerance}.csv"
        )
        exportor.run()