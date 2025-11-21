from abc import ABC, abstractmethod
import os
from typing import List

class SourceManager: 
    '''
    Manages the location and retrivies relevant information about the source data for a dataset. 
    '''
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def list_subsets(self, split: str) -> List[str]:
        split_path = os.path.join(self.base_path, split)
        if not os.path.exists(split_path):
            return []
        return [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

    def subset_path(self, split: str, subset: str) -> str:
        return os.path.join(self.base_path, split, subset)
    
class ShardManager:
    '''
    Manages the location of precomputed features for a dataset. 
    '''
    def __init__(self, base_path: str):
        self.base_path = base_path
 
    def locate_shard(self, sliding_style: str, split: str, subset: str, shard_type: str) -> str:
        result = os.path.join(self.base_path, sliding_style, split, subset, shard_type)
        return result
    
    def locate_base(self, sliding_style: str, split: str, subset: str = "") -> str:
        return os.path.join(self.base_path, sliding_style, split, subset)
    