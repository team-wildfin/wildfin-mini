from abc import ABC, abstractmethod
import os
from typing import List

class ShardManager(ABC):
    '''
    manages the relative path of a shard given the sliding style, shard id. 
    '''
    def __init__(self, base_path: str): 
        self.base_path = base_path

    @abstractmethod
    def list_subsets(self, split: str) -> List[str]:
        '''
        Given the split, list the subset paths in the split. 
        '''
        pass

    @abstractmethod
    def subset_path(self, split: str, subset: str) -> str:
        '''
        Given the split and subset name, return the path to the subset.
        '''
        pass

    @abstractmethod
    def locate_precomputed(self, split: str, subset: str, sliding_style: str, shard_type: str) -> str:
        '''
        Given the dataset name, split name, sliding style name, and subset name, return the path to the shard.
        '''
        pass



class NestedFSShardManager(ShardManager):
    '''
    A simple implementation of ShardManager that assumes a nested file system structure.
    The path is constructed as: {dataset}/{split}/{sliding_style}/{subset}
    '''
    def list_subsets(self, split: str) -> List[str]:
        split_path = os.path.join(self.base_path, split)
        if not os.path.exists(split_path):
            return []
        return [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

    def subset_path(self, split: str, subset: str) -> str:
        return os.path.join(self.base_path, split, subset)

    
    def locate_precomputed(self, split: str, subset: str, sliding_style: str, shard_type: str) -> str:
        result = os.path.join(self.base_path, split, subset, sliding_style, shard_type)
        if not os.path.exists(result): 
            raise FileNotFoundError(f"Precomputed path does not exist: {result}")
        return result