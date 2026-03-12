from vision_bench.management.query import query_pending_evaluations
from config.management.matcher import CORALCAM_TRAINING, CORALCAM_EVAL
from config.experiments.eccv import ECCV_CORALCAM_VJEPA2_ATTENTION

if __name__ == "__main__":
    pending_eval = query_pending_evaluations(
        train_matcher=CORALCAM_TRAINING, 
        eval_matcher=CORALCAM_EVAL, 
        experiments=ECCV_CORALCAM_VJEPA2_ATTENTION,
        rerun=False
    )
    print(pending_eval)