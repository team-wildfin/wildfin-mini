import subprocess
import os
import yaml
from submission import get_slurm_submission_command
from fish_benchmark.utils import setup_logger

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
    'frames_w_temp', 
    # 'sliding_window', 
    # 'sliding_window_w_temp', 
    # 'sliding_window_w_stride', 
    # 'fix_patched_512',
]
SAMPLERS = [
    'random', 
    'balanced'
]
OUTPUT_BASE = os.path.join('logs', 'train')
os.makedirs(OUTPUT_BASE, exist_ok=True)
PARALLEL = False
FULLTUNE = False
model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))

logger = setup_logger(
    'train', 
    os.path.join(OUTPUT_BASE, 'train.log'), 
    console=(not PARALLEL), 
    file=True
)

def get_wrap_cmd(model, classifier, pooling, dataset, sliding_style, sampler):
    return (
        f'python training/head.py '
        f'--classifier {classifier} --pooling {pooling} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model} --sampler {sampler} ' 
    ) if not FULLTUNE else (
        f'python training/fulltune.py '
        f'--classifier {classifier} --pooling {pooling} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model} --sampler {sampler} '
    )

def main():
    for DATASET in DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for MODEL in MODELS:
                for POOLING in POOLINGS: 
                    for CLASSIFIER in CLASSIFIERS:
                        for SAMPLER in SAMPLERS: 
                            if not SLIDING_STYLE in dataset_config[DATASET]['splits']['train']['sliding_styles']: continue
                            if not SLIDING_STYLE in model_config[MODEL]['sliding_styles']: continue
                            wrap_cmd = get_wrap_cmd(MODEL, CLASSIFIER, POOLING, DATASET, SLIDING_STYLE, SAMPLER)
                            OUTPUT_DIR = os.path.join(OUTPUT_BASE, DATASET, SLIDING_STYLE, MODEL, POOLING, CLASSIFIER, SAMPLER)
                            submission_name = f"{MODEL}_{CLASSIFIER}_{POOLING}_{DATASET}_{SLIDING_STYLE}"
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