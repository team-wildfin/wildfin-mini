import os
import subprocess
import yaml
from fish_benchmark.utils.general import setup_logger
from submission import get_slurm_submission_command
from config.datasets import CORALCAM, FISHFOLLOW
# Example config values (replace with loading from a file if needed)

SLIDING_STYLES = [
    # "frames", 
    # "frames_w_temp", 
    # "sliding_window", 
    "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "sliding_window_ti8",
    # "fix_patched_512", 
    # "test_frames", 
    # "test_sliding_window", 
    # "test_sliding_window_ti8",
    # "test_fix_patched_512",
]
PARALLEL = False
SAVE_INPUT = True

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
    for dataset in [CORALCAM, FISHFOLLOW]:
        for split in dataset.splits: 
            for ss_name in set(SLIDING_STYLES).intersection(set(split.get_sliding_style_names())):
                root_dir = os.path.join(dataset.path, split.name)
                dest_root_dir = os.path.join(dataset.precomputed_path, ss_name, split.name)
                for subset in os.listdir(root_dir):
                    assert(os.path.isdir(os.path.join(root_dir, subset))), f"Subset path {subset} is not a directory"
                    source = os.path.join(root_dir, subset)
                    input_dest = os.path.join(dest_root_dir, subset, 'inputs')
                    label_dest = os.path.join(dest_root_dir, subset, 'labels')
                    output_dir = os.path.join('logs', 'slide_window', dataset.name, ss_name, split.name, subset)
                    os.makedirs(output_dir, exist_ok=True)
                    wrap_cmd = get_wrap_cmd(source, input_dest, label_dest, subset, dataset.name, ss_name)
                    submission_name = f"{dataset.name}_{ss_name}_{split.name}_{subset}"
                    command = get_slurm_submission_command(
                            submission_name, output_dir, wrap_cmd, gpu_count=0
                        ) if PARALLEL else wrap_cmd
                    logger.info(f"Running command for {dataset.name}_{ss_name}_{split.name}_{subset} with command: {command}")
                    try: 
                        subprocess.run(command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error running command for {submission_name}: {e}")
            
if __name__ == '__main__':
    main()