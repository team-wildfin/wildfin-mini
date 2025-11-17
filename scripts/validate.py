import av
import yaml
import os
from fish_benchmark.utils.general import setup_logger
from config.maps.model_sliding_style import MODEL_SLIDING_STYLE
from config.data.sliding_styles import SLIDING_STYLES
from config.data.datasets import CORALCAM, FISHFOLLOW
from fish_benchmark.typing.types import LocalDataset, Split, SlidingStyle
from fish_benchmark.execution.validator import Validator

DATASETS = [
    CORALCAM,
    FISHFOLLOW  
]
FEATURE_EXTRACTORS = [
    # 'dino', 
    # 'dino_large', 
    # 'videomae', 
    # 'resnet50', 
    'dinov3_large',
    # 'vjepa2'
]  # or 'clip', etc.
SAVE_DIR = 'data/validation/reports'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
logger = setup_logger(
    "validate_sliding_style",
    console=True,
    file=False,
)
if __name__ == "__main__":
    validator = Validator(
        datasets=DATASETS,
        sliding_styles=SLIDING_STYLES,
    ).set_root(SAVE_DIR)
    validator.run(models=FEATURE_EXTRACTORS)