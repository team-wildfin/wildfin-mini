from typing import Literal
from pydantic import BaseModel, model_validator, Field
from typing import Optional, List, Union, Annotated
from config.models.models import ModelConfig
import yaml


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
    patch_type: Literal["relative", "absolute"]
    patch_h: int
    patch_w: int
    
    @model_validator(mode="after")
    def check_sliding_style(self):
        assert self.window_size % self.samples_per_window == 0, f"window_size {self.window_size} should be a factor of samples_per_window {self.samples_per_window}"
        assert self.tolerance_region <= (self.window_size - 1)//2, f"tolerance_region {self.tolerance_region} should be less than or equal to window_size {self.window_size//2}"
        if self.data_ndim == 3: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"
        return self

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

WeightConfig = Annotated[
    Union[UniformConfig, InverseConfig, FocalLossConfig],
    Field(discriminator='weight_method')
]
