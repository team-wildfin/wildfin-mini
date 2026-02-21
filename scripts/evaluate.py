import logging
from vision_bench.management.wandb_matcher import WandbRunMatcher
from config.experiments.testing import DINOV3_BASE_MEAN
from config.experiments.eccv import ECCV_CORALCAM, ECCV_FISHFOLLOW, ECCV_FISHFOLLOW_DINOV3_BASE, ECCV_FISHFOLLOW_VIDEOMAE_LARGE
from vision_bench.utils.general import setup_logger
from vision_bench.execution.evaluator import Evaluator
from config.main import MODEL_CHECKPOINT_DIR, EVAL_RESULTS_DIR
from config.management.matcher import CORALCAM_TRAINING, CORALCAM_EVAL, FISHFOLLOW_EVAL, FISHFOLLOW_TRAINING

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)
if __name__ == "__main__":
    evaluator = Evaluator(
        experiments = ECCV_FISHFOLLOW_VIDEOMAE_LARGE, 
        train_matcher = FISHFOLLOW_TRAINING,
        eval_matcher = FISHFOLLOW_EVAL,
        parallel = False,
        logger = logger, 
        avoid_reruns = True
    )
    evaluator.run()