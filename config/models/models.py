from pydantic import BaseModel
from typing import Literal, Optional

class ModelConfig(BaseModel):
    name: str
    input_ndim: int
    output_ndim: int
    
class BackBoneConfig(ModelConfig):
    architecture: Literal['cnn', 'transformer']
    hidden_size: int