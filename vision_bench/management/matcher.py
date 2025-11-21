from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Dict
from vision_bench.typing.experiment import Experiment
from vision_bench.typing.types import RunState


class Matcher(ABC): 
    '''
    A general class to run any function using wandb runs from the source project and output to the destination project.
    It can detect if the runs 
    '''

    @abstractmethod
    def match(self, experiments: List[Experiment], states: List[RunState] = ["finished"]) -> Dict[str, str]:
        """
        Get matched run IDs for the given experiments.
        If non_duplicate is True, it will filter out runs that have already been evaluated.
        """
        pass
    
    @abstractmethod
    def match_by_train_id(self, train_ids: List[str], states: List[RunState] = ["finished"]) -> List[str]: 
        """
        Returns the set of evaluation run IDs that correspond to any of the given training run IDs.
        In most cases the train ids should be of the same experiment. 
        """
        pass

    @abstractmethod
    def get_latest(self, run_ids: List[str]) -> str: 
        pass

    @abstractmethod
    def match_config(self, run_config: dict, reference: Experiment) -> bool:
        pass 

    @abstractmethod
    def get_artifact(self, run_id: str, artifact_name: str):
        pass

    @abstractmethod
    def get_run_config(self, run_id: str) -> dict:
        pass