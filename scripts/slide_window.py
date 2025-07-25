import os
import subprocess
import yaml
from fish_benchmark.utils import setup_logger
from submission import get_slurm_submission_command

# Example config values (replace with loading from a file if needed)
TARGETS = [
    "coralcam", 
    "fishfollow"
]
SLIDING_STYLES = [
    # "frames", 
    "frames_w_temp", 
    # "sliding_window", 
    # "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "fix_patched_512", 
    # "test_frames", 
    # "test_sliding_window", 
    # "test_fix_patched_512",
]
PARALLEL = False
SAVE_INPUT = True

config = yaml.safe_load(open("config/actual/dataset.yml", "r"))
logger = setup_logger(
    name = 'slide_window',
    log_file = 'logs/slide_window.log', 
    console = False,
    file = True,
)

def get_wrap_cmd(source, input_dest, label_dest, subset, dataset, sliding_style):
    return (
        f'python data/action/slide_window.py '
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
                    output_dir = os.path.join('logs', 'slide_window', DATASET, SLIDING_STYLE, SPLIT, SUBSET)
                    os.makedirs(output_dir, exist_ok=True)
                    wrap_cmd = get_wrap_cmd(SOURCE, INPUT_DEST, LABEL_DEST, SUBSET, DATASET, SLIDING_STYLE)
                    submission_name = f"{DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET}"
                    command = get_slurm_submission_command(
                            submission_name, output_dir, wrap_cmd, gpu_count=0
                        ) if PARALLEL else wrap_cmd
                    logger.info(f"Running command for {DATASET}_{SLIDING_STYLE}_{SPLIT}_{SUBSET} with command: {command}")
                    try: 
                        subprocess.run(command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error running command for {submission_name}: {e}")
            
if __name__ == '__main__':
    main()