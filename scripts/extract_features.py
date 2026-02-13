"""
extracts features from precomputed inputs
"""
import os
import torch
from vision_bench.utils.general import setup_logger
from config.data.datasets import CORALCAM
from vision_bench.execution.feature_extractor import FeatureExtractor
from vision_bench.execution.validator import Validator
from vision_bench.execution.preprocessor import Preprocessor
from config.main import VALIDATION_REPORTS_DIR
from config.experiments.eccv import ECCV_EXPS, ECCV_CORALCAM, ECCV_FISHFOLLOW
from config.data.datasets import DATASETS
from config.data.sliding_styles import SLIDING_STYLES

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = os.path.join("logs", "extract_features")
if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT, exist_ok=True)
logger = setup_logger(
    "extract_features",
    os.path.join(OUT_ROOT, "extract_fishfollow_features.log"),
    console=True,
    file=False,
    level="INFO"
)

if __name__ == "__main__":
    for EXP in ECCV_CORALCAM:
        dataset = DATASETS[EXP.dataset]
        sliding_style = SLIDING_STYLES[EXP.sliding_style]
        validator = Validator(
            root_path=VALIDATION_REPORTS_DIR,
            logger=logger)

        if not EXP.fulltune: 
            #only tuning the classification head hence need to extract features first
            validator.run(
                dataset=dataset, 
                sliding_style = sliding_style, 
                model = EXP.backbone
            )
            FeatureExtractor(
                validator=validator,
                logger=logger, 
                parallel=True
            ).run(dataset=dataset, 
                sliding_style=sliding_style, 
                model=EXP.backbone
                ) 
        else: 
            #need to extract the input frames
            validator.run(
                dataset=dataset, 
                sliding_style = sliding_style, 
                model = 'inputs'
            )
            Preprocessor(
                validator=validator,
                logger=logger
            ).run(dataset=dataset, sliding_style=sliding_style, save_input=True)