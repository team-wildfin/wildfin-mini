import av
import yaml
import os
from vision_bench.utils.general import setup_logger
from config.data.sliding_styles import SLIDING_STYLES
from config.data.datasets import CORALCAM
from config.main import VALIDATION_REPORTS_DIR
from vision_bench.execution.validator import Validator

FEATURE_EXTRACTORS = [
    "dinov3_base",
]  # or 'clip', etc.
if not os.path.exists(VALIDATION_REPORTS_DIR):
    os.makedirs(VALIDATION_REPORTS_DIR, exist_ok=True)
logger = setup_logger(
    "validate_sliding_style",
    console=True,
    file=False,
)
if __name__ == "__main__":
    Validator(
        datasets=[CORALCAM],
        sliding_styles=SLIDING_STYLES,
        root_path=VALIDATION_REPORTS_DIR,
    ).run(models=FEATURE_EXTRACTORS)