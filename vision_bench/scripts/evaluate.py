import argparse
import wandb
from vision_bench.model.litmodule import LitBinaryClassifierModule
from vision_bench.data.builder import DatasetBuilder
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import lightning as L
import json
from config.data.sliding_styles import SLIDING_STYLES    
import glob
from vision_bench.typing.experiment import Experiment, Evaluation
from config.data.datasets import DATASETS
from config.maps.sliding_style_test import TEST_SLIDING_STYLES
from vision_bench.management.manager import ShardManager
from config.management.matcher import MATCHERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--train_matcher_id", type=str, required=True, help="WandB run ID of the training run to evaluate")
    parser.add_argument("--eval_matcher_id", type=str, required=True, help="WandB run ID of the evaluation run to log results to")
    parser.add_argument("--run", type=str, required=True, help="WandB run ID")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_matcher = MATCHERS[args.train_matcher_id]
    eval_matcher = MATCHERS[args.eval_matcher_id]

    local_ckpt_file = glob.glob(os.path.join(train_matcher.local_artifact_dir, args.run, 'best*.ckpt'))[0]
    ckpt_file = train_matcher.get_artifact(
        local_path = local_ckpt_file,
        remote_path = f"model-{args.run}:latest"
    )
    
    train_config = Experiment.model_validate(train_matcher.get_run_config(args.run))
    test_sliding_style = TEST_SLIDING_STYLES[train_config.sliding_style] # name of the test sliding style
    config = Evaluation.model_validate(
        train_config.model_dump() |
        {
        "test_sliding_style": test_sliding_style,
        "training_run_id": args.run,
        "training_entity": train_matcher.entity,
        "training_project": train_matcher.project
        }
    )
    #define testing config
    tags_keys = [
        'dataset', 
        'sliding_style',
        'backbone',
        'pooling',
        'classifier',
        'sampler',
        'test_sliding_style',
    ]

    wandb_logger = WandbLogger(
        project=eval_matcher.project,    
        entity=eval_matcher.entity,
        save_dir="./logs",
        tags = [v for k, v in config.items() if k in tags_keys] + (["fulltune"] if config.fulltune else []), 
        config=config.model_dump(),
    )
    dataset = DATASETS[config.dataset]
    shard_manager = ShardManager(base_path=dataset.precomputed_path)
    test_dataset = DatasetBuilder(
        path=shard_manager.locate_base(config.test_sliding_style, "test"), 
        dataset=dataset,
        sliding_style=SLIDING_STYLES[config.test_sliding_style],
        input_transform=None,
        precomputed=True,
        feature_model=config.backbone if not config.fulltune else None # if not precomoputed, the feature model is the downloaded backbone
    ).build()
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=wandb_logger.experiment.config["batch_size"],
        shuffle=False,
        num_workers=7,
        pin_memory=True,
    )

    lit_module = LitBinaryClassifierModule.load_from_checkpoint(ckpt_file)
    lit_module.freeze()
    lit_module.eval()
    lit_module.to(device)
    trainer = L.Trainer(logger=wandb_logger, log_every_n_steps= 50)
    trainer.test(lit_module, test_dataloader)

    probs = torch.stack(lit_module.prob_list).cpu()
    targets = torch.stack(lit_module.target_list).cpu()
    output_dict = {
        "probs": probs.tolist(),
        "targets": targets.tolist()
    }

    if not os.path.exists(eval_matcher.local_artifact_dir):
        os.makedirs(eval_matcher.local_artifact_dir, exist_ok=True)
    json_path = os.path.join(eval_matcher.local_artifact_dir, f"{wandb_logger.experiment.id}.json")
    with open(json_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"Saved JSON output to {json_path}")

    # Create and log W&B artifact
    artifact = wandb.Artifact(
        name=f"test_metrics_{wandb_logger.experiment.id}.json",
        type="metrics",
        description="Model test probabilities and targets",
    )
    artifact.add_file(json_path)
    wandb_logger.experiment.log_artifact(artifact)