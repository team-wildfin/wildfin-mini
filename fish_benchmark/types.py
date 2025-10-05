from typing import Literal
from pydantic import BaseModel, model_validator
from typing import Optional, List
from config.models.models import ModelConfig
import yaml

config = yaml.safe_load(open("config/models.yml", "r"))

Weight = Literal["uniform", "inverse", "focal_loss"]
LocalDataset = Literal['coralcam', 'fishfollow']
Sampler = Literal['random', 'balanced']
Metric = Literal['val_mAP']
Optimizer = Literal['adam']
LabelType = Literal['onehot']


class SlidingStyle(BaseModel):
    name: str
    window_size: int
    tolerance_region: int
    samples_per_window: int 
    step_size: int
    data_ndim: int
    shuffle: bool
    patch_type: Literal["relative", "absolute"]
    patch_h: int
    patch_w: int
    temporal_sample_interval: int
    MAX_BUFFER_SIZE: int
    
    @model_validator(mode="after")
    def check_sliding_style(self):
        assert self.window_size % self.samples_per_window == 0, f"window_size {self.window_size} should be a factor of samples_per_window {self.samples_per_window}"
        assert self.tolerance_region <= (self.window_size - 1)//2, f"tolerance_region {self.tolerance_region} should be less than or equal to window_size {self.window_size//2}"
        if self.data_ndim == 3: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"

class Split(BaseModel):
    name: Literal['train', 'val', 'test']
    sliding_styles: List[SlidingStyle]
    def get_sliding_style_names(self) -> List[str]:
        return [style.name for style in self.sliding_styles]

class LocalDataset(BaseModel):
    name: str
    doi: Optional[str] = None
    path: str
    precomputed_path: str
    categories: List[str]
    label_type: LabelType
    splits: List[Split]

# SlidingStyle = Literal[
#     'frames', 
#     'frames_w_temp', 
#     'sliding_window', 
#     'sliding_window_w_temp', 
#     'sliding_window_w_stride', 
#     'fix_patched_512', 
#     'sliding_window_ti8',
# ]

Backbone = Literal['dino', 'dino_large', 'videomae', 'resnet50']
Classifier = Literal['mlp']
Pooling = Literal['mean', 'attention']

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
    '''
    The configuration for a training experiment. Purpose to be displayed on wandb. 
    '''
    #id of the train config
    id: str

    #data configs
    dataset: str
    sliding_style: str

    #model configs
    backbone: str
    pooling: Pooling
    classifier: Classifier
    
    #training configs
    weight_config: WeightConfig
    weight_decay: float
    sampler: Sampler
    fulltune: bool
    freeze_backbone: bool
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


class Evaluation(Experiment):
    test_sliding_style: str
    training_run_id: str
    training_entity: str
    training_project: str