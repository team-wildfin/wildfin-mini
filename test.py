# import torch
# import random
# from torch.utils.data import Sampler

# class MultiLabelBalancedSampler(Sampler):
#     def __init__(self, label_tensor, num_samples_per_class=None):
#         """
#         Args:
#             label_tensor (torch.Tensor): [N, C] binary tensor, where 1 indicates the sample belongs to class C.
#             num_samples_per_class (int, optional): Number of samples to draw per class group. 
#                 If None, uses the maximum class group size.
#         """
#         self.label_tensor = label_tensor  # [N, C]
#         self.num_classes = label_tensor.size(1)
#         self.class_to_indices = [[] for _ in range(self.num_classes)]

#         # Build mapping: class_id -> list of indices containing that class
#         for idx in range(label_tensor.size(0)):
#             for class_id in torch.nonzero(label_tensor[idx]).squeeze(-1).tolist():
#                 self.class_to_indices[class_id].append(idx)

#         # Decide how many samples to draw from each class group
#         self.max_per_class = max(len(indices) for indices in self.class_to_indices)
#         self.samples_per_class = num_samples_per_class or self.max_per_class

#         # Oversample each group
#         for class_id in range(self.num_classes):
#             indices = self.class_to_indices[class_id]
#             if len(indices) < self.samples_per_class:
#                 extra = random.choices(indices, k=self.samples_per_class - len(indices))
#                 self.class_to_indices[class_id].extend(extra)

#     def __iter__(self):
#         indices = [-1] * self.num_classes
#         current_class = 0

#         while indices[current_class] < self.samples_per_class - 1:
#             indices[current_class] += 1
#             yield self.class_to_indices[current_class][indices[current_class]]
#             current_class = (current_class + 1) % self.num_classes

#     def __len__(self):
#         return self.num_classes * self.samples_per_class

# import torch
# from collections import Counter
# from torch.utils.data import Dataset, DataLoader

# # === Dummy multi-label data ===
# label_tensor = torch.tensor([
#     [1, 0, 0],  # sample 0: class 0 and 2
#     [0, 1, 0],  # sample 1: class 1
#     [1, 0, 0],  # sample 2: class 0 and 1
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [0, 0, 1],  # sample 3: class 2
#     [1, 0, 0],  # sample 4: class 0, 1, 2
# ])  # shape: [5, 3]

# # === Dummy dataset ===
# class DummyDataset(Dataset):
#     def __init__(self, N):
#         self.data = list(range(N))
#     def __getitem__(self, idx):
#         return self.data[idx]
#     def __len__(self):
#         return len(self.data)

# dataset = DummyDataset(len(label_tensor))

# # === Import your sampler (assumed defined) ===
# sampler = MultiLabelBalancedSampler(label_tensor)

# # === Wrap in DataLoader ===
# loader = DataLoader(dataset, batch_size=1, sampler=sampler)

# # === Collect sampled indices ===
# sampled_indices = [x.item() for batch in loader for x in batch]

# # === Count total occurrences of each data point ===
# sample_counter = Counter(sampled_indices)

# print("Sample count per index:")
# for idx in range(len(label_tensor)):
#     print(f"Sample {idx}: {sample_counter[idx]}")

# # === Count number of times each class was sampled ===
# class_counts = torch.zeros(label_tensor.shape[1], dtype=torch.int32)

# for idx in sampled_indices:
#     class_counts += label_tensor[idx]

# print("\nClass counts from sampled data:")
# for c in range(label_tensor.shape[1]):
#     print(f"Class {c}: {class_counts[c].item()}")

# # === Expected output: each class should appear sampler.samples_per_class times ===
# print(f"\nExpected per-class sample count: {sampler.samples_per_class}")
# print(f"Total samples drawn: {len(sampled_indices)}")


# import torch
# from torch.utils.data import Dataset, DataLoader, Sampler
# import time
# from collections import Counter
# import random
# # === Simulate imbalanced multi-label data ===
# NUM_SAMPLES = 100_000
# NUM_CLASSES = 50
# LABELS_PER_SAMPLE = 1

# # Imbalanced label distribution (exponential decay)
# torch.manual_seed(42)
# decay = torch.exp(-torch.linspace(0, 5, NUM_CLASSES))  # sharper exponential decay
# class_probs = decay / decay.sum()
# label_tensor = torch.zeros(NUM_SAMPLES, NUM_CLASSES, dtype=torch.uint8)
# for i in range(NUM_SAMPLES):
#     sampled_classes = torch.multinomial(class_probs, LABELS_PER_SAMPLE, replacement=False)
#     label_tensor[i, sampled_classes] = 1

# # === Realistic dataset that returns (data, label) ===
# class MultiLabelDataset(Dataset):
#     def __init__(self, data_tensor, label_tensor):
#         self.data_tensor = data_tensor
#         self.label_tensor = label_tensor
#     def __getitem__(self, idx):
#         return self.data_tensor[idx], self.label_tensor[idx]
#     def __len__(self):
#         return len(self.data_tensor)

# data_tensor = torch.randn(NUM_SAMPLES, 16)  # dummy 16-dim data
# dataset = MultiLabelDataset(data_tensor, label_tensor)

# class MultiLabelBalancedSampler(Sampler):
#     def __init__(self, dataset, max_samples_per_class=1000):
#         """
#         Args:
#             dataset: A dataset with a `.label_tensor` attribute of shape [N, C] (multi-hot labels).
#             max_samples_per_class (int): Maximum number of samples to draw for each class per epoch.
#         """
#         if not hasattr(dataset, "label_tensor"):
#             raise ValueError("Dataset must have a `.label_tensor` attribute of shape [N, num_classes]")
        
#         self.label_tensor = dataset.label_tensor
#         self.num_classes = self.label_tensor.shape[1]
#         self.max_samples_per_class = max_samples_per_class
#         self.class_to_indices = [[] for _ in range(self.num_classes)]

#         # Vectorized: collect all (sample_idx, class_idx) pairs
#         idx_class_pairs = torch.nonzero(self.label_tensor, as_tuple=False)
#         for class_id in range(self.num_classes):
#             self.class_to_indices[class_id] = (
#                 idx_class_pairs[idx_class_pairs[:, 1] == class_id][:, 0].tolist()
#             )

#     def __iter__(self):
#         sampled_indices = []

#         for class_id in range(self.num_classes):
#             indices = self.class_to_indices[class_id]
#             if len(indices) >= self.max_samples_per_class:
#                 # Randomly sample without replacement
#                 sampled = random.sample(indices, self.max_samples_per_class)
#             else:
#                 # Oversample with replacement
#                 sampled = random.choices(indices, k=self.max_samples_per_class)
#             sampled_indices.extend(sampled)

#         # Shuffle to avoid class order bias
#         random.shuffle(sampled_indices)
#         return iter(sampled_indices)

#     def __len__(self):
#         return self.max_samples_per_class * self.num_classes


# # === Simulate actual data pipeline ===
# sampler = MultiLabelBalancedSampler(dataset)
# loader = DataLoader(dataset, sampler=sampler, batch_size=256, num_workers=0)

# # === Count number of times each class was sampled ===
# class_counts = torch.zeros(NUM_CLASSES, dtype=torch.int32)

# start_time = time.time()

# for data_batch, label_batch in loader:
#     # Sum over batch dimension to get per-class presence counts
#     class_counts += label_batch.sum(dim=0).int()

# elapsed = time.time() - start_time

# # === Print class sample counts ===
# print(f"\n✅ Pipeline ran in {elapsed:.2f} seconds")
# print("Sample counts per class:")
# for c in range(NUM_CLASSES):
#     print(f"Class {c:2d}: {class_counts[c].item():6d} (expected ~{sampler.max_samples_per_class})")

# # === Print actual dataset imbalance as reference ===
# print("\nOriginal class frequencies:")
# original_class_freq = dataset.label_tensor.sum(dim=0)
# for c in range(NUM_CLASSES):
#     print(f"Class {c:2d}: {original_class_freq[c].item():6d}")

# NUM_EPOCHS = 50
# coverage_tracker = [set() for _ in range(dataset.label_tensor.shape[1])]

# for epoch in range(NUM_EPOCHS):
#     sampler = MultiLabelBalancedSampler(dataset, max_samples_per_class=1000)
#     for idx in sampler:
#         class_ids = torch.nonzero(dataset.label_tensor[idx]).squeeze(-1).tolist()
#         for class_id in class_ids:
#             coverage_tracker[class_id].add(idx)

# # === Report coverage for each class ===
# print(f"\n📊 Coverage after {NUM_EPOCHS} epochs:")
# for class_id in range(dataset.label_tensor.shape[1]):
#     total = sum(dataset.label_tensor[:, class_id]).item()
#     covered = len(coverage_tracker[class_id])
#     coverage_pct = 100.0 * covered / total if total > 0 else 0.0
#     print(f"Class {class_id:2d}: {covered:5d} / {total:5d} ({coverage_pct:5.1f}%)")

# # Optional: classes not fully covered
# uncovered = [i for i, s in enumerate(coverage_tracker) if len(s) < sum(dataset.label_tensor[:, i])]
# if uncovered:
#     print(f"\n⚠️ Classes not fully covered after {NUM_EPOCHS} epochs: {uncovered}")
# else:
#     print("\n✅ All class members have been sampled at least once.")

import os
import torch
import pandas as pd

train_dir = "your_path"

# List all video subdirectories (e.g., GX017045_95, etc.)
for video_dir in sorted(os.listdir(train_dir)):
    full_video_path = os.path.join(train_dir, video_dir)
    if not os.path.isdir(full_video_path):
        continue  # skip files

    # Path to the TSV label file
    tsv_path = os.path.join(full_video_path, "labels", f"{video_dir}.tsv")
    
    if not os.path.exists(tsv_path):
        print(f"⚠️ Missing: {tsv_path}")
        continue

    # Load and convert label tensor
    df = pd.read_csv(tsv_path, sep='\t', header=None)
    label_tensor = torch.tensor(df.values, dtype=torch.uint8)

    print(f"{video_dir} → label_tensor shape: {label_tensor.shape}")
