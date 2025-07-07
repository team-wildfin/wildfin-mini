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
from transformers import VideoMAEModel, CLIPVisionModel, AutoModel, Swinv2Model, TimesformerModel
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoImageProcessor, AutoProcessor
import torch
from fish_benchmark.data.preprocessors import TorchVisionPreprocessor
from transformers import AutoConfig
import yaml
from abc import ABC, abstractmethod



class HasInputNDims(ABC):
    def get_input_ndim(self):
        assert hasattr(self, 'input_ndim'), (
            f"{self.__class__.__name__} must have an 'input_ndim' attribute"
        )
        """Return the number of expected input dimensions."""
        return self.input_ndim

'''
Pooling classes
'''
class BasePooler(HasInputNDims):
    '''
    A pooler pools [batch, tokens, dim] to [batch, dim]
    '''
    def __init__(self):
        super().__init__()
        self.input_ndim = 2

    def get_input_ndim(self):
            return self.input_ndim
    
    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: [batch, tokens, dim]
        Returns:
            [batch, dim]
        """
        pass

class MeanPooling(BasePooler, nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.mean(dim=self.dim)
    
class MaxPooling(BasePooler, nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim).values

class AttentionPooling(BasePooler, nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.query_token.expand(B, -1, -1)  # [B, 1, D]
        x = self.norm(x)
        attn_out, _ = self.attn(q, x, x)
        return attn_out.squeeze(1)  # [B, D]

'''
Classifier classes
'''
class BaseClassifier(HasInputNDims):
    def __init__(self):
        super().__init__()
        self.input_ndim = 1

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: [batch, dim]
        Returns:
            [batch, num_classes]
        """
        pass

class MLP(BaseClassifier, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        assert num_layers >= 2, "MLP must have at least 2 layers"
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class Linear(BaseClassifier, nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

'''
Backbone classes
'''
class BackBoneModel(nn.Module, HasInputNDims):
    def __init__(self, model_name):
        super().__init__()
        self.model = self.get_pretrained_model(model_name)
        self.config = self.model.config
        self.ioconfig = yaml.safe_load(open("config/models.yml", "r"))[model_name]
        self.input_ndim = self.ioconfig['input_ndim']
    
    def get_pretrained_model(self, model_name):
        if model_name == 'clip':
            return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        elif model_name == 'dino':
            return AutoModel.from_pretrained('facebook/dinov2-base')
        elif model_name == 'dino_large':
            return AutoModel.from_pretrained('facebook/dinov2-large')
        elif model_name == 'videomae':
            return VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        elif model_name == 'timesformer':
            return TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        elif model_name == 'swinv2':
            return Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
    def forward(self, x):
        out = self.model(x).last_hidden_state
        return out

 
class PoolerFactory:
    def __init__(self, pooler_type, dim, hidden_size=None):
        self.pooler_type = pooler_type
        self.dim = dim
        self.hidden_size = hidden_size
    
    def build(self):
        if self.pooler_type == 'mean':
            return MeanPooling(dim=self.dim)
        elif self.pooler_type == 'max':
            return MaxPooling(dim=self.dim)
        elif self.pooler_type == 'attention':
            assert self.hidden_size is not None, "Attention pooling requires hidden_size"
            return AttentionPooling(embed_dim=self.hidden_size)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooler_type}")

class ClassifierFactory:
    def __init__(self, classifier_type, input_dim, output_dim):
        self.classifier_type = classifier_type
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self):
        if self.classifier_type == 'mlp':
            return MLP(input_dim=self.input_dim, hidden_dim=512, output_dim=self.output_dim, num_layers=2)
        elif self.classifier_type == 'linear':
            return Linear(self.input_dim, self.output_dim)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")


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

def get_input_transform(model_name, do_resize = None):
    if model_name == 'dino':
        processor = TorchVisionPreprocessor()
        return processor

    elif model_name == 'dino_large':
        processor = TorchVisionPreprocessor()
        return processor
    
    elif model_name == 'videomae':
        processor = TorchVisionPreprocessor()
        return processor
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class ComposedModel(nn.Module):
    def __init__(self, backbone, pooling, classifier):
        super().__init__()
        self.freeze = True
        self.backbone = backbone
        self.pooling = pooling
        self.classifier = classifier
        self.set_freeze_pretrained(self.freeze)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.classifier(x)
        return x
    
    def set_freeze_pretrained(self, freeze):
        self.freeze = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        return self


class ModelBuilder():
    def __init__(self, backbone: str = None, pooling: str = None, classifier: str = None, hidden_size: int = None, aggregator: str = None):
        self.backbone = backbone
        self.pooling = pooling
        self.classifier = classifier
        self.hidden_size = hidden_size
        self.aggregator = aggregator
        self.classifier_input_dim = None
        self.classifier_output_dim = None

    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
        return self

    def get_hidden_size(self):
        return self.hidden_size
    
    def set_backbone(self, backbone):
        self.backbone = backbone
        self.hidden_size = BackBoneModel(backbone).config.hidden_size
        return self
    
    def set_pooling(self, pooling):
        self.pooling = pooling
        return self
    
    def set_aggregator(self, aggregator):
        self.aggregator = aggregator
        return self
    
    def set_classifier(self, classifier, input_dim, output_dim):
        self.classifier = classifier
        self.classifier_input_dim = input_dim
        self.classifier_output_dim = output_dim
        return self

    @classmethod
    def from_config(cls, config):
        # Extract the model parameters from the config
        backbone = config.get("backbone", None)
        pooling = config.get("pooling", None)
        classifier = config.get("classifier", None)
        hidden_size = config.get("hidden_size", None)
        aggregator = config.get("aggregator", None)
        return cls(backbone, pooling, classifier, hidden_size, aggregator)

    def build(self):
        #dimension check
        if self.classifier and self.backbone: assert self.classifier_input_dim == self.hidden_size, f"Classifier input dimension {self.classifier_input_dim} does not match backbone hidden size {self.hidden_size}"
        BACKBONE = BroadcastableModule(BackBoneModel(self.backbone)) if self.backbone else nn.Identity()
        POOLING = BroadcastableModule(PoolerFactory(self.pooling, dim=1, hidden_size=self.hidden_size).build()) if self.pooling else nn.Identity()
        CLASSIFIER = BroadcastableModule(ClassifierFactory(self.classifier, self.classifier_input_dim, self.classifier_output_dim).build()) if self.classifier else nn.Identity()
        AGGREGATOR = BroadcastableModule(PoolerFactory(self.aggregator, dim=1).build()) if self.aggregator else nn.Identity()
        return nn.Sequential(
            BACKBONE,
            POOLING,
            CLASSIFIER,
            AGGREGATOR
        )