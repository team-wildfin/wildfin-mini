from config.management.matcher import (CORALCAM_TRAINING, 
                                       CORALCAM_EVAL, 
                                       FISHFOLLOW_EVAL, 
                                       FISHFOLLOW_TRAINING)
from config.experiments.eccv import ECCV_CORALCAM_VJEPA2_ATTENTION
from vision_bench.management.query import query_trained, query_evaluated
RUN_ID = "erzk5bf8"

if __name__ == "__main__":
    run_config = CORALCAM_TRAINING.get_run_config(RUN_ID)
    print("run config:", run_config)
    print(f"Experiment: {ECCV_CORALCAM_VJEPA2_ATTENTION[0]}")
    matched1 = CORALCAM_TRAINING.match_config(run_config, ECCV_CORALCAM_VJEPA2_ATTENTION[0])
    matched2 = CORALCAM_TRAINING.match_config(run_config, ECCV_CORALCAM_VJEPA2_ATTENTION[1])
    print(f"Matched {ECCV_CORALCAM_VJEPA2_ATTENTION[0].id}: {matched1}")
    print(f"Matched {ECCV_CORALCAM_VJEPA2_ATTENTION[1].id}: {matched2}")
    trained = query_trained(CORALCAM_TRAINING, ECCV_CORALCAM_VJEPA2_ATTENTION)
    print(f"Trained experiments: {trained}")