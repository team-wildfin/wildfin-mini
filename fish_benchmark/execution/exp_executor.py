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
                 parallel: bool = False, 
                 local_artifact_dir: str = "./artifacts"):
        """
        Args: 
            experiments: A list of Experiment objects to run.
            train_matcher: A WandbRunMatcher object to match training runs.
            eval_matcher: A WandbRunMatcher object to match evaluation runs. Optional.
            logger: A logging.Logger object to log messages. If None, a default logger will be created.
            parallel: Whether to run experiments in parallel using Slurm. Default is False.
            local_artifact_dir: The local directory to store artifacts. If an artifact is not found
                locally, it will be downloaded from W&B and stored here.
        """
        self.experiments = experiments
        self.train_matcher = train_matcher
        self.eval_matcher = eval_matcher
        self.parallel = parallel
        self.logger = logger or logging.getLogger(__name__)
        self.local_artifact_dir = local_artifact_dir
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass