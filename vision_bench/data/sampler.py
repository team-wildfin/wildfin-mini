"""
sampler.py

Contains custom PyTorch Sampler(s) for balanced sampling in multi-label classification tasks.

Includes:
- MultiLabelBalancedSampler: Ensures each class is sampled equally per epoch, supporting multi-hot label tensors.
"""

import torch
from torch.utils.data import Sampler
import random

class MultiLabelBalancedSampler(Sampler):
    """
    A PyTorch Sampler that balances sampling across all classes in a multi-label dataset.

    Args:
        dataset: A dataset with a `.label_tensor` attribute of shape [N, C] (multi-hot labels).
        max_samples_per_class (int): Maximum number of samples to draw for each class per epoch.

    This sampler ensures that each class is sampled equally, oversampling or undersampling as needed.
    """
    def __init__(self, dataset, max_samples_per_class=1000):
        if not hasattr(dataset, "label_tensor"):
            raise ValueError("Dataset must have a `.label_tensor` attribute of shape [N, num_classes]")
        self.label_tensor = dataset.label_tensor
        # Augment the label tensor with a dummy class for the "no label" case
        self.label_tensor = torch.cat([
            self.label_tensor,
            (self.label_tensor.sum(dim=1, keepdim=True) == 0).float()
        ], dim=1)
        self.num_classes = self.label_tensor.shape[1]
        self.max_samples_per_class = max_samples_per_class
        self.class_to_indices = [[] for _ in range(self.num_classes)]
        # Vectorized: collect all (sample_idx, class_idx) pairs
        idx_class_pairs = torch.nonzero(self.label_tensor, as_tuple=False)
        for class_id in range(self.num_classes):
            self.class_to_indices[class_id] = (
                idx_class_pairs[idx_class_pairs[:, 1] == class_id][:, 0].tolist()
            )
        self.non_zero = self.label_tensor.sum(dim=0) > 0

    def __iter__(self):
        sampled_indices = []
        for class_id in range(self.num_classes):
            indices = self.class_to_indices[class_id]
            if not self.non_zero[class_id]:
                continue
            if len(indices) >= self.max_samples_per_class:
                # Randomly sample without replacement
                sampled = random.sample(indices, self.max_samples_per_class)
            else:
                # Oversample with replacement
                sampled = random.choices(indices, k=self.max_samples_per_class)
            sampled_indices.extend(sampled)
        # Shuffle to avoid class order bias
        random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        return self.max_samples_per_class * self.non_zero.sum().item() 