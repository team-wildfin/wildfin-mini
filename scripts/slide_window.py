
from config.maps.sliding_style_test import TEST_NAME
from vision_bench.utils.general import setup_logger
from config.data.datasets import CORALCAM
from config.data.sliding_styles import SLIDING_STYLES
from config.data.datasets import DATASETS
from vision_bench.execution.preprocessor import Preprocessor
from config.experiments.eccv import ECCV_EXPS
from vision_bench.typing.types import LocalDataset, SlidingStyle
from itertools import chain
from typing import Set, Tuple

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
    dset_ss_pairs: Set[Tuple[LocalDataset, SlidingStyle]] = set(chain.from_iterable(
        [
        [(DATASETS[EXP.dataset], SLIDING_STYLES[EXP.sliding_style]), 
         (DATASETS[EXP.dataset], SLIDING_STYLES[TEST_NAME[EXP.sliding_style]])]
        for EXP in ECCV_EXPS
        ]
    ))
    for dset, ss in dset_ss_pairs:
        Preprocessor(
            logger=logger,
            parallel=PARALLEL
        ).run(
            dataset=dset,
            sliding_style=ss,
            save_input=False)