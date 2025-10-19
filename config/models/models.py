from pydantic import BaseModel
from typing import Literal, Optional

class ModelConfig(BaseModel):
    name: str
    input_ndim: int
    output_ndim: int
    
class BackBoneConfig(ModelConfig):
    architecture: Literal['cnn', 'transformer']
    crop_size: tuple[int, int]
    hidden_size: int