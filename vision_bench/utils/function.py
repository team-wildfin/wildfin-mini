import torch
from typing import Callable, Any, Union, List, Dict

class Pipe:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __or__(self, func):
        result = func(*self.args, **self.kwargs)

        # If result is a tuple, treat it as *args for the next step
        if isinstance(result, tuple):
            self.args = result
            self.kwargs = {}
        else:
            self.args = (result,)
            self.kwargs = {}

        return self

    def result(self):
        if len(self.args) == 1 and not self.kwargs:
            return self.args[0]
        return self.args if self.args else self.kwargs

portal: Callable[[torch.Tensor, torch.Tensor], Any] = lambda x, y: Pipe(x, y)