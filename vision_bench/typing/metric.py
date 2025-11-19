from abc import ABC, abstractmethod
import torch 
from typing import Any, Callable

class ToleranceMetric(ABC):     
    def __init__(self, tolerance: int = 0):
        self.tolerance = tolerance

    @abstractmethod
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> Any:
        pass

class MetricPipeline(ABC): 
    @abstractmethod
    def __preprocess__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def __compute__(self, preds: torch.Tensor, targets: torch.Tensor) -> Any:
        pass

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> Any:
        preds, targets = self.__preprocess__(preds, targets)
        return self.__compute__(preds, targets)

Metric = Callable[[torch.Tensor, torch.Tensor], Any]
