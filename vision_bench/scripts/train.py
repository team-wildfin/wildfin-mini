import argparse
import json
import logging
import os
import sys
import yaml
import torch
import lightning as L

from typing import Union
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from vision_bench.model.models import ModelBuilder
from vision_bench.data.builder import DatasetBuilder
from vision_bench.data.sampler import MultiLabelBalancedSampler
from vision_bench.model.litmodule import LitBinaryClassifierModule
from vision_bench.typing.experiment import Experiment  # Your TrainConfig class
from config.data.datasets import DATASETS
from config.models.backbones import BACKBONE_CONFIGS
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.utils.artifact import log_best_model, log_latest_model
from vision_bench.management.manager import ShardManager
from config.management.matcher import MATCHERS

logger = logging.getLogger(__name__)

def load_config(path: Union[str, Path]) -> Experiment:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) if str(path).endswith((".yml", ".yaml")) else json.load(f)
    return Experiment(**raw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher_id", type=str, required=True, help="WandB run ID of the training run to evaluate")
    parser.add_argument("--config", type=str, required=True, help="Path to TrainConfig YAML or JSON")
    args = parser.parse_args()

    matcher = MATCHERS[args.matcher_id]
    config: Experiment = load_config(args.config)
    backbone_config = BACKBONE_CONFIGS[config.backbone]
    consumed_ndim = backbone_config.input_ndim - backbone_config.output_ndim
    sliding_style_config = SLIDING_STYLES[config.sliding_style]
    dataset = DATASETS[config.dataset]
    aggregator = (
        "max"
        if sliding_style_config.data_ndim - consumed_ndim - 1 > 1
        else None
    )

    tags = [
        config.dataset,
        config.sliding_style,
        config.backbone,
        config.pooling,
        config.classifier,
        config.sampler,
        "fulltune" if config.fulltune else "head",
        config.weight_config.weight_method,
    ]
    if config.weight_config.weight_method == "focal_loss":
        tags.append(
            f"gamma_{config.weight_config.focal_loss_gamma}_alpha_{config.weight_config.focal_loss_alpha}"
        )

    wandb_logger = WandbLogger(
        project=matcher.project,
        entity=matcher.entity,
        save_dir="./logs",
        log_model="best",
        tags=tags,
        config=config.model_dump(),  # Use model_dump to get a dict representation
    )

    logger.info("Loading train data...")
    shard_manager = ShardManager(base_path=dataset.precomputed_path)
    train_dataset = DatasetBuilder(
        path=shard_manager.locate_base(config.sliding_style, "train", config.train_subset or ""),
        dataset=dataset,
        sliding_style=sliding_style_config,
        input_transform=None,
        precomputed=True,
        feature_model=None if config.fulltune else config.backbone,
    ).build()

    logger.info("Loading val data...")
    val_dataset = DatasetBuilder(
        path=shard_manager.locate_base(config.sliding_style, "val", config.val_subset or ""),
        dataset=dataset,
        sliding_style=sliding_style_config,
        input_transform=None,
        precomputed=True,
        feature_model=None if config.fulltune else config.backbone,
    ).build()

    logger.info("Data loaded.")
    if config.sampler == "balanced":
        train_sampler = MultiLabelBalancedSampler(
            train_dataset, max_samples_per_class=config.max_samples_per_class
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset,
            num_samples=config.max_samples_per_class * (len(train_dataset.categories) + 1),
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        num_workers=7,
        shuffle=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=7, shuffle=False
    )

    if config.fulltune:
        logger.info("Building full fine-tuning model...")
        model = (
            ModelBuilder.build(
                backbone_name=config.backbone,
                pooler_name=config.pooling,
                classifier_name=config.classifier,
                aggregator_name=aggregator,
                hidden_size=None,
                output_dim=len(dataset.categories),
                freeze_backbone=config.freeze_backbone
            )
        )
    else:
        logger.info("Building classifier head on frozen features...")
        hidden_size = BACKBONE_CONFIGS[config.backbone].hidden_size
        model = (
            ModelBuilder.build(
                backbone_name=None,
                pooler_name=None,
                classifier_name=config.classifier,
                aggregator_name=aggregator,
                hidden_size=hidden_size,
                output_dim=len(train_dataset.categories),
                freeze_backbone=False
            )
        )

    best_ckpt = ModelCheckpoint(
        monitor=config.monitor,
        save_top_k=1,
        mode="max",
        dirpath=f"{matcher.local_artifact_dir}/{wandb_logger.experiment.id}",
        filename="best-{epoch:02d}-{val_mAP:.2f}",
    )

    latest_ckpt = ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        dirpath=f"{matcher.local_artifact_dir}/{wandb_logger.experiment.id}",
        filename="latest",
    )

    lit_module = LitBinaryClassifierModule(
        model,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_config=config.weight_config.model_dump(),
    )
    lit_module.set_root_path(dataset.path)

    tqdm_disable = not sys.stdout.isatty()
    logger.info(f"Are we in an interactive terminal? {not tqdm_disable}")
    trainer = L.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[best_ckpt, latest_ckpt],
        check_val_every_n_epoch=5,
    )
    trainer.fit(lit_module, train_dataloader, val_dataloader)

    try:
        log_best_model(best_ckpt, wandb_logger.experiment)
    except Exception as e:
        logger.error(f"Error logging best model: {e}")
    try:
        log_latest_model(latest_ckpt, wandb_logger.experiment)
    except Exception as e:
        logger.error(f"Error logging latest model: {e}")

if __name__ == "__main__":
    main()