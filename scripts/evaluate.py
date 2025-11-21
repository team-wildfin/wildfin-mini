import logging
from vision_bench.management.wandb_matcher import WandbRunMatcher
from config.experiments.testing import DINOV3_BASE_MEAN
from vision_bench.utils.general import setup_logger
from vision_bench.execution.evaluator import Evaluator
from config.main import MODEL_CHECKPOINT_DIR, EVAL_RESULTS_DIR
from config.management.matcher import CORALCAM_TRAINING, CORALCAM_EVAL

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
EVAL_PROJECT = f"{TRAINING_PROJECT}_eval"
PARALLEL = False

if __name__ == "__main__":
    evaluator = Evaluator(
        experiments = DINOV3_BASE_MEAN, 
        train_matcher = CORALCAM_TRAINING,
        eval_matcher = CORALCAM_EVAL,
        parallel = PARALLEL,
        logger = logger, 
        avoid_reruns = False
    )
    evaluator.run()