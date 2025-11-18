from vision_bench.data.shard_manager import ShardManager, NestedFSShardManager
from vision_bench.typing.types import LocalDataset
from typing import List, Type
import logging
from abc import ABC, abstractmethod
from config.logging.loggers import get_console_logger

class ShardExecutor: 
    def __init__(self, 
                 datasets: List[LocalDataset], 
                 sliding_styles: List[str], 
                 parallel: bool = False, 
                 logger: logging.Logger = get_console_logger(), 
                 ShardManager: Type[ShardManager] = NestedFSShardManager):
        """
        Initialize the Preprocessor with datasets, sliding styles, and other options.
        Args: 
            datasets (List[LocalDataset]): List of datasets to process.
            sliding_styles (List[str]): List of sliding styles to apply.
            parallel (bool): Whether to run in parallel using SLURM.
            save_input (bool): Whether to save the input data after processing.
            logger (logging.Logger): Logger for logging messages.
            ShardManager (Type[ShardManager]): Class for managing data shards.
        """
        self.datasets = datasets
        self.sliding_styles = sliding_styles
        self.parallel = parallel
        self.logger = logger
        self.ShardManager = ShardManager

    @abstractmethod
    def run(self, *args, **kwargs): 
        pass 