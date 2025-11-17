from fish_benchmark.typing.experiment import Experiment
from typing import List, Optional, Callable
import logging
from abc import ABC, abstractmethod
from fish_benchmark.management.matcher import Matcher
from fish_benchmark.management.wandb_matcher import WandbRunMatcher


class ExperimentExecutor(ABC): 
    """
    A general class to run experiments that are matched with runs. 
    """
    def __init__(self, 
                 experiments: List[Experiment], 
                 train_matcher: WandbRunMatcher, 
                 eval_matcher: Optional[WandbRunMatcher] = None, 
                 logger: Optional[logging.Logger] = None, 
                 parallel: bool = False):
        self.experiments = experiments
        self.train_matcher = train_matcher
        self.eval_matcher = eval_matcher
        self.parallel = parallel
        self.logger = logger or logging.getLogger(__name__)
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass