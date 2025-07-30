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
    
#aggregate metrics
Metric = Callable[[torch.Tensor, torch.Tensor], Any] #2d tensors with the same shape
Filter = Callable[[torch.Tensor, torch.Tensor], torch.Tensor] #expects full targets, returns a mask 
f1_micro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_f1_score, average='micro', num_labels = y.shape[1]) | tensor_to_basic).result()
f1_macro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_f1_score, average='macro', num_labels = y.shape[1]) | tensor_to_basic).result()
precision_micro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_precision, average='micro', num_labels = y.shape[1]) | tensor_to_basic).result()
precision_macro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_precision, average='macro', num_labels = y.shape[1]) | tensor_to_basic).result()
recall_micro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_recall, average='micro', num_labels = y.shape[1]) | tensor_to_basic).result()
recall_macro: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_recall, average='macro', num_labels = y.shape[1]) | tensor_to_basic).result()
mAP: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_average_precision, average='macro', num_labels = y.shape[1]) | tensor_to_basic).result()
acc: Metric = lambda x, y: (Pipe(x, y) | (lambda x, y: ((x > 0.5) == y).float().mean()) | tensor_to_basic).result()

#per class metrics
mAP_per_class: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_average_precision, average=None, num_labels=y.shape[1]) | tensor_to_basic).result()
f1_per_class: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_f1_score, average=None, num_labels=y.shape[1]) | tensor_to_basic).result()
precision_per_class: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_precision, average=None, num_labels=y.shape[1]) | tensor_to_basic).result()
recall_per_class: Metric = lambda x, y: (Pipe(x, y) | partial(multilabel_recall, average=None, num_labels=y.shape[1]) | tensor_to_basic).result()
positive_per_class: Metric = lambda _, y: (Pipe(_, y) | (lambda _, y: y.sum(dim=0).int()) | tensor_to_basic).result()

def union(lst: List[Dict]) -> Dict:
    """
    Union of a list of dictionaries, merging entries by taking the union of keys.
    Later rows override only if earlier value is empty or missing.
    """
    return reduce(lambda a, b: a | b, lst, {})