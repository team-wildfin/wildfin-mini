
from vision_bench.utils.general import setup_logger
from config.data.datasets import CORALCAM
from vision_bench.execution.preprocessor import Preprocessor
# Example config values (replace with loading from a file if needed)
SLIDING_STYLES = [
    "frames_w_temp", 
    "sliding_window_w_temp", 
    "test_frames", 
    "test_sliding_window", 
]
PARALLEL = False
SAVE_INPUT = True

logger = setup_logger(
    name = 'slide_window',
    log_file = 'logs/slide_window.log', 
    console = False,
    file = True,
)
if __name__ == "__main__": 
    Preprocessor(
        datasets=[CORALCAM],
        sliding_styles=SLIDING_STYLES,
        parallel=PARALLEL,
    ).run(save_input=SAVE_INPUT)