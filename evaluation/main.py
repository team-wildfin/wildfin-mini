import argparse
import wandb
from fish_benchmark.litmodule import LitBinaryClassifierModule
from fish_benchmark.data.dataset import DatasetBuilder
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import lightning as L
import json
from config.sliding_styles import SLIDING_STYLES    
import glob
from fish_benchmark.typing.experiment import Experiment, Evaluation
from config.datasets import DATASETS
from config.maps.backbone_preprocessors import PREPROCESSORS

eval_config = yaml.safe_load(open('config/eval.yml', 'r'))
checkpoint_path = yaml.safe_load(open('config/training.yml', 'r'))['checkpoint_path']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
TEST_METRIC_DIR = os.path.join('logs', 'test_metrics')
os.makedirs(TEST_METRIC_DIR, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--run", type=str, required=True, help="WandB run ID")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    #load the artifact
    try: 
        api = wandb.Api()
        artifact = api.artifact(f"{args.entity}/{args.project}/model-{args.run}:latest", type="model")
        artifact_dir = artifact.download()
        print(f"Artifact downloaded to {artifact_dir}")
        ckpt_files = glob.glob(os.path.join(artifact_dir, "*.ckpt"))
        assert len(ckpt_files) == 1, f"Expected exactly one .ckpt file in {artifact_dir}"
        ckpt_file = ckpt_files[0]
    except Exception as e:
        #load from ./checkpoints/<run_id>/best<something>.ckpt
        print(f"Failed to download artifact: {e}")
        print(f"Trying to load from local path")
        ckpt_pattern = os.path.join(checkpoint_path, args.run, 'best*.ckpt')
        ckpt_files = glob.glob(ckpt_pattern)
        assert len(ckpt_files) == 1, f"Expected exactly one checkpoint file matching 'best*.ckpt' in {ckpt_pattern}"
        ckpt_file = ckpt_files[0]
        print(f"Loaded checkpoint from {ckpt_file}")
        
    training_run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run}")
    train_config = Experiment.model_validate(training_run.config)
    test_sliding_style = eval_config[train_config.sliding_style] # name of the test sliding style
    config = Evaluation.model_validate(
        train_config.model_dump() |
        {
        "test_sliding_style": test_sliding_style,
        "training_run_id": args.run,
        "training_entity": args.entity,
        "training_project": args.project
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
        project=f'{config.dataset}_eval',    
        entity="fish-benchmark",
        save_dir="./logs",
        tags = [v for k, v in config.items() if k in tags_keys] + (["fulltune"] if config.fulltune else []), 
        config=config.model_dump(),
    )
    dataset = DATASETS[config.dataset]
    test_data_dir = os.path.join(dataset.precomputed_path, config.test_sliding_style, 'test') 
    test_dataset = DatasetBuilder(
        path=test_data_dir, 
        dataset=dataset,
        sliding_style=SLIDING_STYLES[config.test_sliding_style],
        transform=None,
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

    json_path = os.path.join(TEST_METRIC_DIR, f"{wandb_logger.experiment.id}.json")
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

#python evaluation/main.py --entity fish-benchmark --project abby --run g5hc3uqy