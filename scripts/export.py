#https://wandb.ai/fish-benchmark/coralcam_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os
from vision_bench.management.wandb_matcher import WandbRunMatcher
from config.experiments.testing import DINOV3_BASE_MEAN
from config.experiments.eccv import ECCV_CORALCAM, ECCV_FISHFOLLOW
import logging
from vision_bench.utils.general import setup_logger
from config.metrics.metrics import * 
from vision_bench.execution.exportor import Exportor
from config.management.matcher import CORALCAM_EVAL, CORALCAM_TRAINING, FISHFOLLOW_EVAL, FISHFOLLOW_TRAINING
from config.data.datasets import CORALCAM, FISHFOLLOW
logger = setup_logger("export", "logs/export.log", console=True, file=True, level=logging.INFO)

# ==== CONFIG ====
LABEL_TOLERANCES = [7]
PARALLEL = False
DOWNLOAD_DIR = "test_metrics"
OUTPUT_PATH = 'wildfins/results/eccv'
os.makedirs(OUTPUT_PATH, exist_ok=True)
ALL_EXPS = ECCV_CORALCAM

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
        Exportor(
            experiments=ALL_EXPS,
            train_matcher=CORALCAM_TRAINING,
            eval_matcher=CORALCAM_EVAL, 
            aggregate_metrics=aggregate_metrics,
            per_class_metrics=per_class_metrics,
            subgroup_mappings=subgroup_mappings[CORALCAM.name], 
            output_base=OUTPUT_PATH,
            output_name=f"{CORALCAM.name}_eccv_label_tolerance_{tolerance}.csv"
        ).run()