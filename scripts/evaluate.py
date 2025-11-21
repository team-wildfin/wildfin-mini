import logging
from vision_bench.management.wandb_matcher import WandbRunMatcher
from config.experiments.testing import DINOV3_BASE_MEAN
from vision_bench.utils.general import setup_logger
from vision_bench.execution.evaluator import Evaluator
from config.experiments.defaults import MISSING_VALUES
from config.main import MODEL_CHECKPOINT_DIR, EVAL_RESULTS_DIR

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
RERUN = False
EVAL_PROJECT = f"{TRAINING_PROJECT}_eval"
PARALLEL = False

if __name__ == "__main__":
    evaluator = Evaluator(
        experiments = DINOV3_BASE_MEAN, 
        train_matcher = WandbRunMatcher(ENTITY, TRAINING_PROJECT, MISSING_VALUES),
        eval_matcher = WandbRunMatcher(ENTITY, EVAL_PROJECT, MISSING_VALUES),
        parallel = PARALLEL,
        logger = logger, 
        model_ckpt_dir = MODEL_CHECKPOINT_DIR, 
        local_artifact_dir = EVAL_RESULTS_DIR
    )
    evaluator.run()