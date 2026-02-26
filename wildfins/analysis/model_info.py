from config.experiments.cvpr import CVPR_EXPS
from config.experiments.eccv import ECCV_EXPS
from fvcore.nn import FlopCountAnalysis, parameter_count
from config.models.backbones import BACKBONE_CONFIGS
import torch
from typing import Callable, List
from vision_bench.model.models import ModelBuilder
from vision_bench.typing.experiment import Experiment
from config.data.datasets import DATASETS 
import yaml
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))

#note: Checkpoints sometimes only the classification head, which is not we want. 
# We want the whole model to calculate FLOPs and parameters. 
# So we need to build the model using the same config as training, and load the checkpoint weights into the model. 
# This way we can ensure that we are calculating FLOPs and parameters for the entire model, not just the classification head.
def get_model_infos(EXPS: List[Experiment], calc_fun: Callable, name: str): 
    results = []
    for exp in EXPS: 
        print(f"Processing experiment {exp.id}...")
        # 2️⃣ Build model using your builder
        model = ModelBuilder.build(
            backbone_name=exp.backbone,
            pooler_name=exp.pooling,
            classifier_name=exp.classifier,
            aggregator_name=None,
            hidden_size=None,
            output_dim=len(DATASETS[exp.dataset].categories),
            freeze_backbone=exp.freeze_backbone,
        )
        backbone_config = BACKBONE_CONFIGS[exp.backbone]
        # backbone_config.crop_size = (224, 224)
        fake_input = (torch.randn(1, 3, *backbone_config.crop_size) 
                    if backbone_config.input_ndim == 3 
                    else torch.randn(1, 16, 3, *backbone_config.crop_size)) 
        exp_result = exp.model_dump() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exp_result[name] = calc_fun(model.to(device), fake_input.to(device))
        results.append(exp_result)
    return results

if __name__ == "__main__":
    EXPS = CVPR_EXPS + ECCV_EXPS
    flops = get_model_infos(EXPS, lambda model, fake_input: FlopCountAnalysis(model, fake_input).total(), "flops")
    parameters = get_model_infos(EXPS, lambda model, fake_input: parameter_count(model)[''], "parameters")
    with open(os.path.join(script_dir, "flops.yaml"), "w") as f:
        yaml.safe_dump(flops, f, sort_keys=False)

    with open(os.path.join(script_dir, "parameters.yaml"), "w") as f:
        yaml.safe_dump(parameters, f, sort_keys=False)