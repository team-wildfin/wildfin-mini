import subprocess
import os
import json
import yaml
from submission import get_slurm_submission_command
from fish_benchmark.utils import setup_logger
from fish_benchmark.types import Weight

#arguments of the file to run
# python training/head.py --classifier mlp --dataset abby --sliding_style frames --model dino
MODELS = [
    # 'dino', 
    # 'dino_large',
    # 'videomae', 
    'resnet50'
]
CLASSIFIERS = [
    'mlp'
]
POOLINGS = [
    'mean', 
    # 'attention'
]
DATASETS = [
    'coralcam', 
    # 'fishfollow'
]
SLIDING_STYLES = [
    'frames', 
    # 'frames_w_temp', 
    # 'sliding_window', 
    # 'sliding_window_w_temp', 
    # 'sliding_window_w_stride', 
    # 'fix_patched_512',
]
SAMPLERS = [
    'random', 
    'balanced'
]

WEIGHT_CONFIGS = [
    {
        'weight_method': 'uniform',
    }
    # {
    #     'weight_method': 'inverse',
    # },
    # {
    #     'weight_method': 'focal_loss',
    #     'focal_loss_gamma': 1.0,
    #     'focal_loss_alpha': 0.5
    # },
    # {
    #     'weight_method': 'focal_loss',
    #     'focal_loss_gamma': 5.0,
    #     'focal_loss_alpha': 0.75
    # },
    # {
    #     'weight_method': 'focal_loss',
    #     'focal_loss_gamma': 2.0,
    #     'focal_loss_alpha': 0.9
    # },
    # {
    #     'weight_method': 'focal_loss',
    #     'focal_loss_gamma': 10.0,
    #     'focal_loss_alpha': 0.05
    # },
    # {
    #     'weight_method': 'focal_loss',
    #     'focal_loss_gamma': 5.0,
    #     'focal_loss_alpha': 0.1
    # },
]


OUTPUT_BASE = os.path.join('logs', 'train')
os.makedirs(OUTPUT_BASE, exist_ok=True)
PARALLEL = False
FULLTUNE = True
model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))

logger = setup_logger(
    'train', 
    os.path.join(OUTPUT_BASE, 'train.log'), 
    console=(not PARALLEL), 
    file=True
)

def get_wrap_cmd(model, classifier, pooling, dataset, sliding_style, sampler, weight_config):
    weight_config_str = f"'{json.dumps(weight_config)}'"
    print(f"Weight config: {weight_config_str}")
    return (
        f'python training/head.py '
        f'--classifier {classifier} --pooling {pooling} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model} --sampler {sampler} --weight_config {weight_config_str} ' 
    ) if not FULLTUNE else (
        f'python training/fulltune.py '
        f'--classifier {classifier} --pooling {pooling} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model} --sampler {sampler} --weight_config {weight_config_str} '
    )

def main():
    for DATASET in DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for MODEL in MODELS:
                for POOLING in POOLINGS: 
                    for CLASSIFIER in CLASSIFIERS:
                        for SAMPLER in SAMPLERS: 
                            for WEIGHT_CONFIG in WEIGHT_CONFIGS:
                                if not SLIDING_STYLE in dataset_config[DATASET]['splits']['train']['sliding_styles']: continue
                                if not SLIDING_STYLE in model_config[MODEL]['sliding_styles']: continue
                                wrap_cmd = get_wrap_cmd(MODEL, CLASSIFIER, POOLING, DATASET, SLIDING_STYLE, SAMPLER, WEIGHT_CONFIG)
                                weight_method = WEIGHT_CONFIG['weight_method']
                                if weight_method == 'focal_loss':
                                    weight_method += f"_gamma_{WEIGHT_CONFIG['focal_loss_gamma']}_alpha_{WEIGHT_CONFIG['focal_loss_alpha']}"
                                OUTPUT_DIR = os.path.join(OUTPUT_BASE, DATASET, SLIDING_STYLE, MODEL, POOLING, CLASSIFIER, SAMPLER, weight_method)
                                submission_name = f"{MODEL}_{CLASSIFIER}_{POOLING}_{DATASET}_{SLIDING_STYLE}_{weight_method}"
                                command = get_slurm_submission_command(
                                    submission_name,
                                    OUTPUT_DIR,
                                    wrap_cmd,
                                    gpu_count=1
                                ) if PARALLEL else wrap_cmd
                                logger.info(f"Running command for {submission_name} with command: {command}")
                                subprocess.run(command, shell=True, check=True)
        
if __name__ == "__main__":
    main()