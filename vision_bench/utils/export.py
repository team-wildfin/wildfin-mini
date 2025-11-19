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
    multilabel_average_precision, 
    binary_average_precision
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
    

def expand_confusion_matrices(matrices: torch.Tensor, names: List[str]) -> Dict[str, List[List[int]]]:
    """
    Expand confusion matrices to a dictionary with names as keys.
    Requires: tensor has shape [num_classes, 2, 2] where num_classes is the number of classes.
    """
    assert matrices.ndim == 3 and matrices.shape[1] == 2 and matrices.shape[2] == 2, \
        "Confusion matrix tensor must have shape [num_classes, 2, 2]"
    expanded = {}
    for i, name in enumerate(names):
        expanded[name] = matrices[i].cpu().numpy().tolist()
    return expanded


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

def union(lst: List[Dict]) -> Dict:
    """
    Union of a list of dictionaries, merging entries by taking the union of keys.
    Later rows override only if earlier value is empty or missing.
    """
    return reduce(lambda a, b: a | b, lst, {})