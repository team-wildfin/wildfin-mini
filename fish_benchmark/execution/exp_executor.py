from fish_benchmark.typing.experiment import Experiment
from typing import List, Optional, Callable
import logging
from abc import ABC, abstractmethod
from fish_benchmark.management.matcher import Matcher


class ExperimentExecutor(ABC): 
    def __init__(self, experiments: List[Experiment], 
                 train_matcher: Matcher, 
                 eval_matcher: Optional[Matcher] = None, 
                 logger: Optional[logging.Logger] = None, 
                 parallel: bool = False):
        self.experiments = experiments
        self.train_matcher = train_matcher
        self.eval_matcher = eval_matcher
        self.parallel = parallel
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass