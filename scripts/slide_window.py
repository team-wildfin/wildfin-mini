import os
import subprocess
import yaml
from vision_bench.utils.general import setup_logger
from vision_bench.utils.submission import get_slurm_submission_command
from config.data.datasets import CORALCAM, FISHFOLLOW
from vision_bench.execution.preprocessor import Preprocessor
# Example config values (replace with loading from a file if needed)
SLIDING_STYLES = [
    # "frames", 
    # "frames_w_temp", 
    # "sliding_window", 
    # "sliding_window_w_temp", 
    # "sliding_window_w_stride", 
    # "sliding_window_ti8",
    # "fix_patched_512", 
    "test_frames", 
    "test_sliding_window", 
    # "test_sliding_window_ti8",
    # "test_fix_patched_512",
]
PARALLEL = True
SAVE_INPUT = True

logger = setup_logger(
    name = 'slide_window',
    log_file = 'logs/slide_window.log', 
    console = False,
    file = True,
)
if __name__ == "__main__": 
    datasets = [CORALCAM, FISHFOLLOW]
    Preprocessor(
        datasets=datasets,
        sliding_styles=SLIDING_STYLES,
        parallel=PARALLEL,
        logger=logger
    ).run(
        save_input=SAVE_INPUT
    )