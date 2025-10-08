from transformers import VideoMAEModel, CLIPVisionModel, AutoModel, Swinv2Model, TimesformerModel, ResNetModel
"""
Model definitions and composition for WildFin.

This module provides:
- Backbone model wrappers for various vision architectures (CLIP, DINO, VideoMAE, etc.)
- Pooling layers (mean, max, attention)
- Classifiers (MLP, linear)
- Factories for building poolers and classifiers
- A ModelBuilder class for flexible model composition
- BroadcastableModule for handling variable input shapes
"""
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoImageProcessor, AutoProcessor
import torch
from fish_benchmark.data.preprocessors import TorchVisionPreprocessor
from transformers import AutoConfig
import yaml
from contextlib import nullcontext
from abc import ABC, abstractmethod
from config.models.backbones import BACKBONE_CONFIGS, BACKBONE_MODULES, PREPROCESSORS, ModelConfig, BackBoneConfig
from config.models.poolers import POOLER_MODULES
from config.models.classifiers import CLASSIFIER_MODULES
import inspect 
from typing import Union, Optional

class FreezableBackbone(nn.Module, ABC):
    def __init__(self, *, freeze: bool = False):
        super().__init__()
        self._frozen = False
        if freeze:
            self.freeze()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, *args, **kwargs):
        ctx = torch.no_grad() if self._frozen else nullcontext()
        with ctx:
            return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class CNN(FreezableBackbone):
    def __init__(self, model_name, freeze: bool = False):
        super().__init__(freeze=freeze)
        self.config = BACKBONE_CONFIGS[model_name]
        self.model = BACKBONE_MODULES[model_name]()
        self.input_ndim = self.config.input_ndim
        assert self.config.hidden_size == self.model.config.hidden_size, (
            f"Model hidden size {self.model.config.hidden_size[-1]} does not match config hidden size {self.config.hidden_size}"
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x).last_hidden_state  # (B, C, H, W)
        B, C, H, W = feats.shape
        return feats.flatten(2).transpose(1, 2)  # (B, H*W, C)

class TransformerModel(FreezableBackbone):
    def __init__(self, model_name: str, *, freeze: bool = False):
        super().__init__(freeze=freeze)
        self.model = BACKBONE_MODULES[model_name]()
        self.config = BACKBONE_CONFIGS[model_name]
        self.input_ndim = self.config.input_ndim
        assert self.config.hidden_size == self.model.config.hidden_size, (
            f"Model hidden size {self.model.config.hidden_size} does not match config hidden size {self.config.hidden_size}"
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state  # (B, L, H)

class BroadcastableModule(nn.Module):
    '''
    A wrapper for models so that they can be broadcasted with multiple batch dimensions. 
    '''
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        assert hasattr(model, 'input_ndim'), (
            f"{model.__class__.__name__} must have an 'input_ndim' attribute"
        )
        self.input_ndim = self.model.input_ndim
        self.chunk_size = 128

    def forward(self, x):
        assert x.ndim >= self.input_ndim + 1, f"Input tensor must have at least {self.input_ndim + 1} dimensions, got {x.ndim}"
        batch_shape = x.shape[: x.ndim - self.input_ndim]
        input_shape = x.shape[-self.input_ndim:]

        flat_x = x.view(-1, *input_shape)
        # Unflatten batch dimensions
        out_list = []
        for i in range(0, flat_x.shape[0], self.chunk_size):
            chunk = flat_x[i : i + self.chunk_size]
            out_chunk = self.model(chunk)
            out_list.append(out_chunk)

        out = torch.cat(out_list, dim=0)
        out_shape = out.shape[1:]  # exclude flattened batch dim
        out = out.view(*batch_shape, *out_shape)
        return out


class ModelBuilder():
    @staticmethod
    def build_backbone(backbone_name, freeze_backbone) -> Union[CNN, TransformerModel]:
        '''
        Builds the backbone model. Side Effect: Updates self.hidden_size.
        '''
        if backbone_name is None:
            return nn.Identity()
        backbone_config = BACKBONE_CONFIGS[backbone_name]
        backbone = (TransformerModel(backbone_config.name, freeze=freeze_backbone)      
                if backbone_config.architecture == 'transformer' 
                else CNN(backbone_config.name, freeze=freeze_backbone))
        return backbone

    @staticmethod
    def build_pooling(pooler, hidden_size) -> nn.Module:
        '''
        Builds the pooling layer.
        '''
        if pooler is None:
            return nn.Identity()
        module_cls = POOLER_MODULES[pooler]
        args = {
            k: v for k, v in {
                'dim': 1, 'hidden_size': hidden_size
            }.items() 
                if k in inspect.signature(module_cls).parameters
        }
        return module_cls(**args)
    
    @staticmethod
    def build_classifier(classifier: str, in_features: str, out_features: str) -> nn.Module:
        if classifier is None:
            return nn.Identity()
        module_cls = CLASSIFIER_MODULES[classifier]
        args = {
            k: v for k, v in {
                'in_features': in_features, 
                'out_features': out_features, 
            }.items() 
                if k in inspect.signature(module_cls).parameters
        }
        return module_cls(**args)

    @staticmethod
    def build(backbone_name: Optional[str], 
              pooler_name: Optional[str], 
              classifier_name: Optional[str], 
              aggregator_name: Optional[str], 
              hidden_size: Optional[int],
              output_dim: int, 
              freeze_backbone: bool) -> nn.Module:
        '''
        Builds the full model.
        '''
        backbone = ModelBuilder.build_backbone(backbone_name, freeze_backbone)
        if not hidden_size: 
            hidden_size = backbone.config.hidden_size
        elif pooler_name is not None or classifier_name is not None: 
            assert backbone is None or hidden_size == backbone.config.hidden_size, (
                f"Provided hidden size {hidden_size} does not match backbone hidden size {backbone.config.hidden_size}"
            )
        pooler = ModelBuilder.build_pooling(pooler_name, hidden_size)
        classifier = ModelBuilder.build_classifier(classifier_name, hidden_size, output_dim)
        aggregator = ModelBuilder.build_pooling(aggregator_name, hidden_size)
        return nn.Sequential(
            backbone,
            pooler,
            classifier,
            aggregator
        )