import csv
import os
from typing import List, Dict, Any, Callable, Union, Optional
import torch
from functools import partial, reduce
import numpy as np

from torchmetrics.functional.classification import (
    multilabel_precision,
    multilabel_recall,
    multilabel_f1_score,
    multilabel_average_precision
)
from abc import ABC, abstractmethod
import torch.nn.functional as F


def load_existing_csv(path: str) -> List[Dict[str, Any]]:
    """Load existing rows from a CSV file if it exists."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def get_all_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Return all unique fieldnames across rows, preserving first-seen order."""
    seen = set()
    ordered_fields = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_fields.append(key)
    return ordered_fields

def fill_missing_fields(rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """In-place fill missing fields with empty string in each row."""
    for row in rows:
        for field in fieldnames:
            if field not in row:
                row[field] = ""

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write all rows with provided fieldnames to CSV."""
    with open(path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update(existing_rows: List[Dict], new_rows: List[Dict], key: str = "run_id") -> List[Dict]:
    """
    Update existing rows with new rows based on a unique key (e.g., 'run_id').
    If a new row has the same key as an existing row, it will update the existing row.
    """
    existing_dict = {row[key]: row for row in existing_rows}
    for new_row in new_rows:
        existing_dict[new_row[key]] = new_row
    return list(existing_dict.values())

def flood_1d(bits, dis):
    res = np.zeros_like(bits)
    last = None
    for i in range(len(bits)):
        if bits[i] == 1:
            last = i
            res[i] = 1
        elif last is not None and i - last <= dis:
            res[i] = 1
    return res

def flood(bits, dis):
    left = flood_1d(bits, dis)
    right = flood_1d(bits[::-1], dis)[::-1]
    return np.logical_or(left, right)

def flood_all_columns(targets, dis):
    return np.apply_along_axis(partial(flood, dis=dis), axis=0, arr=targets)


def tensor_to_basic(tensor: torch.Tensor) -> Union[float, List[float], List[int]]:
    '''
    Convert a tensor to a basic type (float, list of floats, or list of ints)
    '''
    if tensor.ndim == 0:
        return tensor.item()
    elif tensor.ndim == 1:
        return tensor.tolist()
    else:
        return tensor.cpu().numpy().tolist()
    
def binary_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class (column-wise) binary confusion matrices: shape [num_classes, 2, 2]
    [
 [[TN, FN],     # class 0
  [FP, TP]],

 [[TN, FN],     # class 1
  [FP, TP]],

 [[TN, FN],     # class 2
  [FP, TP]]
]

    """
    # Binarize predictions using threshold 0.5
    preds = (preds >= 0.5).int()
    targets = targets.int()

    num_classes = preds.shape[1]
    conf_matrices = torch.zeros((num_classes, 2, 2), dtype=torch.int)

    for i in range(num_classes):
        p = preds[:, i]
        t = targets[:, i]
        for pred_val in (0, 1):
            for true_val in (0, 1):
                conf_matrices[i, pred_val, true_val] = ((p == pred_val) & (t == true_val)).sum()

    return conf_matrices




class Pipe:
    def __init__(self, *args):
        self.args = args

    def __or__(self, func):
        result = func(*self.args)

        # If result is a tuple, treat it as *args for the next step
        if isinstance(result, tuple):
            self.args = result
        else:
            self.args = (result,)

        return self

    def result(self):
        if len(self.args) == 1:
            return self.args[0]
        return self.args if self.args else None
    
Metric = Callable[[torch.Tensor, torch.Tensor], Any]

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


def flood_1d(x: torch.Tensor, tolerance: int) -> torch.Tensor:
    """
    Dilate/flood a binary tensor along the first dimension (n) by `tolerance`.
    Each 1 in x[i, j] will flood ±tolerance rows in the same column j.

    Args:
        x (torch.Tensor): Binary tensor of shape (n, d)
        tolerance (int): Radius of flooding region (>= 0)

    Returns:
        torch.Tensor: Flooded binary tensor of same shape as x
    """
    if tolerance <= 0:
        return x

    # Treat each column independently, so swap n and d
    x = x.transpose(0, 1).unsqueeze(1).float()  # (d,1,n)
    kernel = torch.ones(1, 1, 2 * tolerance + 1, device=x.device)
    flooded = F.conv1d(x, kernel, padding=tolerance)
    flooded = (flooded > 0).squeeze(1).to(torch.int)
    return flooded.transpose(0, 1)  # back to (n,d)

class Recall(MetricPipeline, ToleranceMetric): 
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        preds = flood_1d(preds, self.tolerance)
        return preds, targets  # matched = recalled ones

class RecallPerClass(Recall): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum(dim=0).float()
        fn = (targets & (~preds)).clamp(min=0).sum(dim=0).float()
        recall = tp / (tp + fn + 1e-8)
        return recall.tolist()

class RecallMacro(ToleranceMetric):
    def __call__(self, preds, targets):
        return torch.tensor(RecallPerClass(self.tolerance)(preds, targets)).mean().item()

class RecallMicro(Recall): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum().float()
        fn = (targets - tp).clamp(min=0).sum().float()
        recall = tp / (tp + fn + 1e-8)
        return recall.item()
    
class Precision(MetricPipeline, ToleranceMetric): 
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        targets = flood_1d(targets, self.tolerance)
        return preds, targets  # matched = true positives

class PrecisionPerClass(Precision):
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum(dim=0).float()
        fp = (preds & (~targets)).clamp(min=0).sum(dim=0).float()
        precision = tp / (tp + fp + 1e-8)
        return precision.tolist()

class PrecisionMacro(ToleranceMetric):
    def __call__(self, preds, targets):
        return torch.tensor(PrecisionPerClass(self.tolerance)(preds, targets)).mean().item()

class PrecisionMicro(Precision): 
    def __compute__(self, preds, targets):
        tp = (preds & targets).sum().float()
        fp = (preds - tp).clamp(min=0).sum().float()
        precision = tp / (tp + fp + 1e-8)
        return precision.item()


# ========= F1 Metrics =========
class F1PerClass(ToleranceMetric):
    def __call__(self, preds, targets):
        precision_per_class = PrecisionPerClass(self.tolerance)(preds, targets)
        recall_per_class = RecallPerClass(self.tolerance)(preds, targets)
        f1_per_class = []
        for p, r in zip(precision_per_class, recall_per_class):
            if p + r == 0:
                f1_per_class.append(0.0)
            else:
                f1_per_class.append(2 * p * r / (p + r))
        return f1_per_class
    
class F1Micro(ToleranceMetric):
    def __call__(self, preds, targets):
        p = PrecisionMicro(self.tolerance)
        r = RecallMicro(self.tolerance)
        precision = p(preds, targets)
        recall = r(preds, targets)
        return 2 * precision * recall / (precision + recall + 1e-8)

class F1Macro(ToleranceMetric):
    def __call__(self, preds, targets):
        p = PrecisionMacro(self.tolerance)
        r = RecallMacro(self.tolerance)
        precision = p(preds, targets)
        recall = r(preds, targets)
        return 2 * precision * recall / (precision + recall + 1e-8)

class PositivePerClass(Metric):
    def __call__(self, preds, targets):
        positives = targets.sum(dim=0).tolist()
        return positives

class Accuracy(MetricPipeline, ToleranceMetric):
    def __preprocess__(self, preds, targets):
        preds = (preds > 0.5).to(torch.int)
        targets = (targets > 0.5).to(torch.int)
        preds_expanded = (flood_1d(preds, self.tolerance) & targets.bool()) | preds.bool()
        targets_expanded = (flood_1d(targets, self.tolerance) & preds.bool()) | targets.bool()
        return preds_expanded, targets_expanded

    def __compute__(self, preds, targets):
        correct = (preds == targets).float().sum()
        total = torch.numel(preds)
        return (correct / total).item()

mAP: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_average_precision, average='macro', num_labels = y.shape[1]) | tensor_to_basic).result()
mAP_per_class: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_average_precision, average=None, num_labels=y.shape[1]) | tensor_to_basic).result()

def union(lst: List[Dict]) -> Dict:
    """
    Union of a list of dictionaries, merging entries by taking the union of keys.
    Later rows override only if earlier value is empty or missing.
    """
    return reduce(lambda a, b: a | b, lst, {})