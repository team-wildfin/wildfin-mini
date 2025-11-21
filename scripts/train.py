import os
from config.experiments.neurips import *
from vision_bench.typing.experiment import Experiment
from vision_bench.management.wandb_matcher import WandbRunMatcher
from vision_bench.utils.general import setup_logger
from config.experiments.testing import DINOV3_BASE_MEAN
from vision_bench.execution.trainer import Trainer
from config.experiments.defaults import MISSING_VALUES
from config.main import MODEL_CHECKPOINT_DIR

PARALLEL = False
OUTPUT_BASE = os.path.join("logs", "train")
ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"

CONFIG_OUT = "generated_configs"
os.makedirs(CONFIG_OUT, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
logger = setup_logger(
    "train", os.path.join(OUTPUT_BASE, "train.log"), console=True, file=True
)

if __name__ == "__main__":
    Trainer(
        DINOV3_BASE_MEAN, 
        WandbRunMatcher(ENTITY, TRAINING_PROJECT, MISSING_VALUES), 
        parallel=PARALLEL, 
        local_artifact_dir=MODEL_CHECKPOINT_DIR,
    ).run()