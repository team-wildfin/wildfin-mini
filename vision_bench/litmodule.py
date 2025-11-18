"""
PyTorch Lightning modules for training and evaluating classification models.

Includes:
- LitBinaryClassifierModule: For multi-label classification
- LitCategoricalClassifierModule: For single-label (categorical) classification
- Logging of metrics (precision, recall, F1, mAP, accuracy)
- Optimizer and scheduler configuration
"""
import torch
import torch.nn.functional as F
import lightning as L
import json
import wandb
from vision_bench.typing.types import Weight
from torch.optim.lr_scheduler import LambdaLR
from data.statistics.pos_frequency import get_pos_freq
from torchmetrics.functional.classification import (
    multilabel_precision,
    multilabel_recall,
    multilabel_f1_score,
    multilabel_average_precision
)
class LitBinaryClassifierModule(L.LightningModule):
    '''
    trains a model multi-label classification task
    '''
    def __init__(self, model, 
                 learning_rate=1e-4, 
                 optimizer = 'adam', 
                 weight_decay = 0.001, 
                 weight_config = {'weight_method': 'uniform'}):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model
        self.prob_list = []
        self.target_list = []
        self.weight_config = weight_config
        self.root_path = None
        

    def set_root_path(self, root_path): 
        self.root_path = root_path
        self.statistics_path = root_path + '/train'
        self.pos_freq = torch.tensor(get_pos_freq(self.statistics_path))
        self.pos_freq = self.pos_freq / sum(self.pos_freq)  # convert to frequencies
        
    def log_additional_metrics(self, prefix, preds, y):
        """
        Logs micro/macro precision, recall, F1, and mAP for multi-label classification.
        
        Args:
            prefix (str): Prefix for logging (e.g., "val" or "test")
            preds (Tensor): shape (B, C), float tensor after sigmoid thresholding (> 0.5)
            y (Tensor): shape (B, C), binary ground truth
        """
        num_classes = preds.shape[1]

        # Precision & Recall (micro and macro)
        prec_micro = multilabel_precision(preds, y, num_labels=num_classes, average='micro')
        prec_macro = multilabel_precision(preds, y, num_labels=num_classes, average='macro')
        rec_micro = multilabel_recall(preds, y, num_labels=num_classes, average='micro')
        rec_macro = multilabel_recall(preds, y, num_labels=num_classes, average='macro')

        self.log(f"{prefix}_precision_micro", prec_micro)
        self.log(f"{prefix}_precision_macro", prec_macro)
        self.log(f"{prefix}_recall_micro", rec_micro)
        self.log(f"{prefix}_recall_macro", rec_macro)

        # F1 scores
        f1_micro = multilabel_f1_score(preds, y, num_labels=num_classes, average='micro')
        f1_macro = multilabel_f1_score(preds, y, num_labels=num_classes, average='macro')

        self.log(f"{prefix}_f1_micro", f1_micro)
        self.log(f"{prefix}_f1_macro", f1_macro)

        f1_per_class = multilabel_f1_score(preds, y, num_labels=num_classes, average=None)
        for i, f1 in enumerate(f1_per_class):
            self.log(f"{prefix}_f1_class_{i}", f1)

        positives_per_class = y.float().sum(dim=0)
        for i, count in enumerate(positives_per_class):
            self.log(f"{prefix}_num_positive_class_{i}", count)

        # mAP
        map = multilabel_average_precision(preds, y, num_labels=num_classes, average='macro')
        self.log(f"{prefix}_mAP", map)

        # Accuracy (strict match)
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_acc", acc)

    def shared_step(self, batch, prefix):
        assert self.root_path is not None, "Root path must be set before training"
        x, y = batch
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        if self.weight_config['weight_method'] == 'uniform':
            loss = F.binary_cross_entropy(probs, y.float(), weight=None)
        elif self.weight_config['weight_method'] == 'inverse':
            inv_weights = 1.0 / (self.pos_freq + 1e-6)
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=inv_weights.to(logits.device))
        elif self.weight_config['weight_method'] == 'focal_loss':
            gamma = self.weight_config['focal_loss_gamma']
            alpha = self.weight_config['focal_loss_alpha']
            probs = torch.sigmoid(logits)
            pt = probs * y + (1 - probs) * (1 - y)
            alpha_t = alpha * y + (1 - alpha) * (1 - y)
            focal_loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt + 1e-8)
            loss = focal_loss.mean()
        else:
            raise ValueError(f"Invalid weight method: {self.weight_config['weight_method']}")
            
        self.log(f'{prefix}_loss', loss)
        preds = (probs > 0.5).float()
        self.log_additional_metrics(prefix, preds, y)
        return {
            "loss": loss,
            "preds": preds,
            "targets": y
        }


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        self.prob_list.extend(probs.detach())   # each item is a list of floats
        self.target_list.extend(y.detach())     # each item is a list of ints
    

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay
        # Linear decay from 1.0 to 0.0
        num_training_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_lambda = lambda current_step: 1.0 - float(current_step) / float(num_training_steps)
        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # called every training step
                "frequency": 1,
                "name": "lr", 
            }
        }

class LitCategoricalClassifierModule(L.LightningModule):
    '''
    subclasses have to have a model component and a classifier component. 
    The labels for the dataset should be a single number indicating the class index.
    '''
    def __init__(self, model, learning_rate=1e-4, optimizer = 'adam'):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def shared_step(self, batch, prefix):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f'{prefix}_loss', loss)
        #train acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f'{prefix}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)