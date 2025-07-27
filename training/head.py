'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from fish_benchmark.models import get_input_transform, ModelBuilder
from fish_benchmark.data.dataset import DatasetBuilder
from fish_benchmark.data.sampler import MultiLabelBalancedSampler
from fish_benchmark.litmodule import LitBinaryClassifierModule
from data.statistics.pos_frequency import get_class_weights
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import sys
from artifact import log_best_model, log_dataset_summary, log_latest_model
import os
import json
# python training/head.py --classifier mlp --dataset abby --sliding_style frames --model dino
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weight_method", default='uniform')
    parser.add_argument("--sliding_style", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--lr", default=.00005)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--shuffle", default=False)
    parser.add_argument("--train_subset", default="")
    parser.add_argument("--val_subset", default="")
    parser.add_argument("--label_type", default='onehot')
    parser.add_argument("--min_ctime", default = '1746331200.0')

    return parser.parse_args()

if __name__ == '__main__':
    dataset_config = yaml.safe_load(open('config/actual/dataset.yml', 'r'))
    sliding_style_config = yaml.safe_load(open('config/sliding_style.yml', 'r'))
    model_config = yaml.safe_load(open('config/models.yml', 'r'))

    args = get_args()
    CLASSIFIER = args.classifier
    POOLING = args.pooling
    DATASET = args.dataset
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    SLIDING_STYLE = args.sliding_style
    SHUFFLE = args.shuffle
    MODEL = args.model
    LABEL_TYPE = args.label_type
    TR_SUBSET = args.train_subset   
    VAL_SUBSET = args.val_subset 
    MIN_CTIME = float(args.min_ctime)
    SAMPLER = args.sampler # 'balanced' or 'random'
    WEIGHT_METHOD = args.weight_method
    MAX_SAMPLES_PER_CLASS = 1000
    MONITOR = 'val_mAP' # 'val_mAP' or 'val_loss'
    consumed_ndim = model_config[MODEL]['input_ndim'] - model_config[MODEL]['output_ndim']
    AGGREGATOR = ('max' if sliding_style_config[SLIDING_STYLE]['data_ndim'] - consumed_ndim - 1 > 1 
                  else None)  # -1 because pooling consumes one dimension. If there are dimensions left, we need to aggregate them 
                            # In the end, we want it to have one dimension which is the number of classes
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    config_dict = {
        #dataset config
        "dataset": DATASET,
        "sliding_style": SLIDING_STYLE,
        #model config
        "backbone": MODEL,
        "pooling": POOLING,
        "classifier": CLASSIFIER,
        "aggregator": AGGREGATOR,
        #training config
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "shuffle": SHUFFLE,
        "sampler": SAMPLER,
        "monitor": MONITOR,
        "fulltune": False, 
        "weight_method": WEIGHT_METHOD,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
    }
    wandb_logger = WandbLogger(
        project=DATASET,    
        entity="fish-benchmark",
        save_dir="./logs",
        log_model="best", 
        tags=[DATASET, SLIDING_STYLE, MODEL, POOLING,CLASSIFIER, SAMPLER, WEIGHT_METHOD],
        config=config_dict,
    )
    print("Loading train data...")
    train_dataset = DatasetBuilder(
        path = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, 'train', TR_SUBSET), 
        dataset_name = DATASET,
        style=SLIDING_STYLE,
        transform=None, 
        precomputed=True, 
        feature_model=MODEL,
        min_ctime=MIN_CTIME,
    ).build()
    print("Loading val data...")
    val_dataset = DatasetBuilder(
        path = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, 'val', VAL_SUBSET), 
        dataset_name = DATASET,
        style=SLIDING_STYLE,
        transform=None, 
        precomputed=True, 
        feature_model=MODEL,
    ).build()
    
    print("Data loaded.")
    train_sampler = (MultiLabelBalancedSampler(train_dataset, max_samples_per_class=wandb_logger.experiment.config["max_samples_per_class"])
               if wandb_logger.experiment.config["sampler"] == 'balanced' else 
               torch.utils.data.RandomSampler(train_dataset, num_samples=MAX_SAMPLES_PER_CLASS * (len(train_dataset.categories) + 1)))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size=wandb_logger.experiment.config["batch_size"], 
        num_workers=7, 
        shuffle=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=wandb_logger.experiment.config["batch_size"], 
        num_workers=7, 
        shuffle=False
    )
    
    # to get hidden size
    hidden_size = ModelBuilder().set_backbone(MODEL).get_hidden_size()
    classifier = (ModelBuilder()
                .set_classifier(CLASSIFIER, input_dim=hidden_size, output_dim=len(train_dataset.categories))
                .set_aggregator(AGGREGATOR)
                .build())
    
    best_ckpt = ModelCheckpoint(
        monitor=MONITOR,
        save_top_k=1,
        mode="max",
        dirpath=f"./checkpoints/{wandb_logger.experiment.id}",
        filename="best-{epoch:02d}-{val_mAP:.2f}",
    )

    # Latest checkpoint (overwrite each epoch)
    latest_ckpt = ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        dirpath=f"./checkpoints/{wandb_logger.experiment.id}",
        filename="latest",
    )
    lit_module = LitBinaryClassifierModule(classifier, 
                                           learning_rate = wandb_logger.experiment.config['learning_rate'], 
                                           optimizer = wandb_logger.experiment.config['optimizer'], 
                                           weight_method=WEIGHT_METHOD)
    lit_module.set_root_path(dataset_config[DATASET]['path'])
    tqdm_disable = not sys.stdout.isatty()
    print(f"Are we in an interactive terminal? {not tqdm_disable}")
    trainer = L.Trainer(max_epochs=wandb_logger.experiment.config['epochs'], 
                        logger=wandb_logger, 
                        log_every_n_steps= 50, 
                        callbacks=[best_ckpt, latest_ckpt], 
                        check_val_every_n_epoch = 5)
    
    trainer.fit(lit_module, train_dataloader, val_dataloader)
    
    try: 
        log_best_model(best_ckpt, wandb_logger.experiment)
    except Exception as e:
        print(f"Error logging best model: {e}")
    try:
        log_latest_model(latest_ckpt, wandb_logger.experiment)
    except Exception as e:
        print(f"Error logging latest model: {e}")
    # log_dataset_summary(train_dataset, run)