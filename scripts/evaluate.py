import logging
from fish_benchmark.management.wandb_matcher import WandbRunMatcher
from config.experiments.cvpr import CVPR_EXPS
from fish_benchmark.utils.general import setup_logger
from fish_benchmark.execution.evaluator import Evaluator
from config.experiments.defaults import MISSING_VALUES

logger = setup_logger("evaluate", console=True, file=False, level=logging.DEBUG)

ENTITY = "fish-benchmark"
TRAINING_PROJECT = "coralcam"
RERUN = False
ALL_EXPS = list(
    filter(
        lambda exp: (
            exp.dataset == TRAINING_PROJECT and 
            exp.backbone in ['vjepa2'] and 
            exp.pooling == 'attention'),
        CVPR_EXPS
    )
)
EVAL_PROJECT = f"{TRAINING_PROJECT}_eval"
PARALLEL = False

if __name__ == "__main__":
    evaluator = Evaluator(
        experiments = ALL_EXPS, 
        train_matcher = WandbRunMatcher(ENTITY, TRAINING_PROJECT, MISSING_VALUES),
        eval_matcher = WandbRunMatcher(ENTITY, EVAL_PROJECT, MISSING_VALUES),
        parallel = PARALLEL,
        logger = logger
    )
    evaluator.run()