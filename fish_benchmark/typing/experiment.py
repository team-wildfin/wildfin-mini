from pydantic import BaseModel, model_validator
from typing import Optional, Literal
from config.models.backbones import MODEL_SLIDING_STYLES
from fish_benchmark.typing.types import Pooling, Classifier, WeightConfig, Sampler, Metric, Optimizer, LabelType

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

        if self.sliding_style not in [ss.name for ss in MODEL_SLIDING_STYLES[self.backbone]]:
            raise ValueError(f"Sliding style {self.sliding_style} is not supported for model {self.backbone}. "
                             f"Supported styles: {[ss.name for ss in MODEL_SLIDING_STYLES[self.backbone]]}")
        return self

class Evaluation(Experiment):
    test_sliding_style: str
    training_run_id: str
    training_entity: str
    training_project: str