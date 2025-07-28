import argparse
import json
import os
import sys
import yaml
import torch
import lightning as L

from typing import Union
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fish_benchmark.models import get_input_transform, ModelBuilder
from fish_benchmark.data.dataset import DatasetBuilder
from fish_benchmark.data.sampler import MultiLabelBalancedSampler
from fish_benchmark.litmodule import LitBinaryClassifierModule
from fish_benchmark.types import TrainConfig  # Your TrainConfig class

from artifact import log_best_model, log_latest_model


def load_config(path: Union[str, Path]) -> TrainConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) if str(path).endswith((".yml", ".yaml")) else json.load(f)
    return TrainConfig(**raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TrainConfig YAML or JSON")
    parser.add_argument("--min_ctime", type=float, default=1746331200.0)
    args = parser.parse_args()

    config: TrainConfig = load_config(args.config)
    min_ctime = args.min_ctime

    dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))
    sliding_style_config = yaml.safe_load(open("config/sliding_style.yml", "r"))
    model_config = yaml.safe_load(open("config/models.yml", "r"))

    consumed_ndim = model_config[config.backbone]["input_ndim"] - model_config[config.backbone]["output_ndim"]
    aggregator = (
        "max"
        if sliding_style_config[config.sliding_style]["data_ndim"] - consumed_ndim - 1 > 1
        else None
    )

    # Match WandB logging to existing runs
    config_dict = {
        "dataset": config.dataset,
        "sliding_style": config.sliding_style,
        "backbone": config.backbone,
        "pooling": config.pooling,
        "classifier": config.classifier,
        "aggregator": aggregator,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "shuffle": config.shuffle,
        "sampler": config.sampler,
        "monitor": config.monitor,
        "fulltune": config.fulltune,
        "weight_config": config.weight_config.model_dump(),
        "max_samples_per_class": config.max_samples_per_class,
    }

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
        project=config.dataset,
        entity="fish-benchmark",
        save_dir="./logs",
        log_model="best",
        tags=tags,
        config=config_dict,
    )

    print("Loading train data...")
    train_dataset = DatasetBuilder(
        path=os.path.join(
            dataset_config[config.dataset]["precomputed_path"],
            config.sliding_style,
            "train",
            config.train_subset or "",
        ),
        dataset_name=config.dataset,
        style=config.sliding_style,
        transform=None,
        precomputed=True,
        feature_model=None if config.fulltune else config.backbone,
        min_ctime=min_ctime,
    ).build()

    print("Loading val data...")
    val_dataset = DatasetBuilder(
        path=os.path.join(
            dataset_config[config.dataset]["precomputed_path"],
            config.sliding_style,
            "val",
            config.val_subset or "",
        ),
        dataset_name=config.dataset,
        style=config.sliding_style,
        transform=None,
        precomputed=True,
        feature_model=None if config.fulltune else config.backbone,
    ).build()

    print("Data loaded.")
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
        print("Building full fine-tuning model...")
        model = (
            ModelBuilder()
            .set_backbone(config.backbone)
            .set_pooling(config.pooling)
            .set_classifier(config.classifier, output_dim=len(train_dataset.categories))
            .set_aggregator(aggregator)
            .build()
        )
    else:
        print("Building classifier head on frozen features...")
        hidden_size = ModelBuilder().set_backbone(config.backbone).get_hidden_size()
        model = (
            ModelBuilder()
            .set_classifier(config.classifier, input_dim=hidden_size, output_dim=len(train_dataset.categories))
            .set_aggregator(aggregator)
            .build()
        )

    best_ckpt = ModelCheckpoint(
        monitor=config.monitor,
        save_top_k=1,
        mode="max",
        dirpath=f"./checkpoints/{wandb_logger.experiment.id}",
        filename="best-{epoch:02d}-{val_mAP:.2f}",
    )

    latest_ckpt = ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        dirpath=f"./checkpoints/{wandb_logger.experiment.id}",
        filename="latest",
    )

    lit_module = LitBinaryClassifierModule(
        model,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_config=config.weight_config.model_dump(),
    )
    lit_module.set_root_path(dataset_config[config.dataset]["path"])

    tqdm_disable = not sys.stdout.isatty()
    print(f"Are we in an interactive terminal? {not tqdm_disable}")
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
        print(f"Error logging best model: {e}")
    try:
        log_latest_model(latest_ckpt, wandb_logger.experiment)
    except Exception as e:
        print(f"Error logging latest model: {e}")


if __name__ == "__main__":
    main()
