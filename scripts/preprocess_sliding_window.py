import os
import subprocess
import yaml
from fish_benchmark.utils import setup_logger
from submission import get_slurm_submission_command

# Example config values (replace with loading from a file if needed)
TARGETS = ["abby", 
           "mike"]
SLIDING_STYLES = [
    # "frames", 
    # "frames_w_temp", 
    # "sliding_window", 
    # "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "fix_patched_512", 
    "test_frames", 
    "test_sliding_window", 
    "test_fix_patched_512",
]
PARALLEL = True
SAVE_INPUT = False

config = yaml.safe_load(open("config/dataset.yml", "r"))
logger = setup_logger(
    name = 'precompute_sliding_window',
    log_file = 'logs/output/precompute_sliding_window.log', 
    console = False,
    file = True,
)

def get_wrap_cmd(source, input_dest, label_dest, subset, dataset, sliding_style):
    return (
        f'python data/action_scripts/preprocess_sliding_window.py '
        f'--source "{source}" --input_dest "{input_dest}" --label_dest "{label_dest}" --id "{subset}" --dataset "{dataset}" '
        f'--save_input {SAVE_INPUT} --sliding_style "{sliding_style}"'
    )

def main():
    for DATASET in TARGETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for SPLIT in list(config[DATASET]['splits'].keys()):
                if SLIDING_STYLE not in config[DATASET]['splits'][SPLIT]['sliding_styles']: continue
                root_dir = os.path.join(config[DATASET]['path'], SPLIT)
                dest_root_dir = os.path.join(config[DATASET]['precomputed_path'], SLIDING_STYLE, SPLIT)
                for SUBSET in os.listdir(root_dir):
                    assert(os.path.isdir(os.path.join(root_dir, SUBSET))), f"Subset path {SUBSET} is not a directory"
                    SOURCE = os.path.join(root_dir, SUBSET)
                    INPUT_DEST = os.path.join(dest_root_dir, SUBSET, 'inputs')
                    LABEL_DEST = os.path.join(dest_root_dir, SUBSET, 'labels')
                    output_dir = os.path.join('logs', 'precompute_sliding_window', DATASET, SLIDING_STYLE, SPLIT, SUBSET)
                    os.makedirs(output_dir, exist_ok=True)
                    # submit a job for each subset
                    if PARALLEL:
                        wrap_cmd = get_wrap_cmd(SOURCE, INPUT_DEST, LABEL_DEST, SUBSET, DATASET, SLIDING_STYLE)
                        name = f"{DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET}"
                        command = get_slurm_submission_command(
                            name, output_dir, wrap_cmd, gpu=0
                        )
                    else:
                        command = get_wrap_cmd(SOURCE, INPUT_DEST, LABEL_DEST, SUBSET, DATASET, SLIDING_STYLE)
                    logger.info(f"Running command for {DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET} with command: {command}")
                    subprocess.run(command, shell=True, check=True)
                
if __name__ == '__main__':
    main()