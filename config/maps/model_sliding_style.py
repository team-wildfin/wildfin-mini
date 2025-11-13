from ..sliding_styles import *
from typing import Dict, List
MODEL_SLIDING_STYLES: Dict[str, List[SlidingStyle]] = {
    "vjepa2": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "videomae": [SLIDING_WINDOW_W_TEMP, TEST_SLIDING_WINDOW],
    "dino": [FRAMES_W_TEMP, TEST_FRAMES],
    "dino_large": [FRAMES_W_TEMP, TEST_FRAMES],
    "dinov3_base": [FRAMES_W_TEMP, TEST_FRAMES],
    "dinov3_large": [FRAMES_W_TEMP, TEST_FRAMES],
    "resnet50": [FRAMES_W_TEMP, FRAMES, TEST_FRAMES],
}