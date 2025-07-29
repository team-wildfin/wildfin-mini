from typing import Literal
from pydantic import BaseModel, model_validator
from typing import Optional
import yaml

config = yaml.safe_load(open("config/models.yml", "r"))

Weight = Literal["uniform", "inverse", "focal_loss"]
Dataset = Literal['coralcam', 'fishfollow']
Sampler = Literal['random', 'balanced']
Metric = Literal['val_mAP']
Optimizer = Literal['adam']
LabelType = Literal['onehot']

SlidingStyle = Literal[
    'frames', 
    'frames_w_temp', 
    'sliding_window', 
    'sliding_window_w_temp', 
    'sliding_window_w_stride', 
    'fix_patched_512'
]

Backbone = Literal['dino', 'dino_large', 'videomae', 'resnet50']
Classifier = Literal['mlp']
Pooling = Literal['mean']

class FocalLossConfig(BaseModel): 
    weight_method: Literal['focal_loss'] = 'focal_loss'
    focal_loss_gamma: float 
    focal_loss_alpha: float

class InverseConfig(BaseModel): 
    weight_method: Literal['inverse'] = 'inverse'

class UniformConfig(BaseModel):
    weight_method: Literal['uniform'] = 'uniform'

WeightConfig = UniformConfig | InverseConfig | FocalLossConfig

class Experiment(BaseModel): 
    #id of the train config
    id: str

    #data configs
    dataset: Dataset
    sliding_style: SlidingStyle

    #model configs
    backbone: Backbone
    pooling: Pooling
    classifier: Classifier
    
    #training configs
    weight_config: WeightConfig
    weight_decay: float
    sampler: Sampler
    fulltune: bool
    epochs: int
    learning_rate: float
    batch_size: int
    optimizer: Optimizer
    shuffle: bool
    monitor: Metric
    label_type: LabelType
    max_samples_per_class: int

    #additional configs
    train_subset: Optional[str] = None
    val_subset: Optional[str] = None

    @model_validator(mode="after")
    def check_sliding_style(self):
        if self.sliding_style not in config[self.backbone]['sliding_styles']:
            raise ValueError(f"Sliding style {self.sliding_style} is not supported for model {self.backbone}. "
                             f"Supported styles: {config[self.backbone]['sliding_styles']}")
        return self