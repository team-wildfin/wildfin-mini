from vision_bench.management.wandb_matcher import WandbRunMatcher
from config.main import MODEL_CHECKPOINT_DIR, EVAL_RESULTS_DIR
from config.experiments.defaults import MISSING_VALUES

ENTITY = 'fish-benchmark'

CORALCAM_TRAINING = WandbRunMatcher(
    entity=ENTITY,
    project='coralcam',
    local_artifact_dir=MODEL_CHECKPOINT_DIR, 
    default_values=MISSING_VALUES
)

CORALCAM_EVAL = WandbRunMatcher(
    entity=ENTITY,
    project='coralcam_eval',
    local_artifact_dir=EVAL_RESULTS_DIR, 
    default_values=MISSING_VALUES
)

MATCHERS = {
    matcher.id: matcher 
    for matcher in globals().values() 
    if isinstance(matcher, WandbRunMatcher)
}