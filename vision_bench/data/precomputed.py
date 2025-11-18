from torch.utils.data import Dataset, IterableDataset, TensorDataset, Sampler
from vision_bench.debug import step_timer
import numpy as np
import torch
import os
from vision_bench.utils.general import get_files_of_type
PROFILE = False

class PrecomputedDataset(Dataset):
    '''
    Dataset mounted on precomputed sliding window clips and labels. 
    Corresponding clips and labels have the same name but live in different folders
    '''
    def __init__(self, path, categories, input_transform=None, label_transform = None, feature_model=None, min_ctime=None):
        '''
        path should be contain 2 subfolders: frames and labels
        '''
        self.label_type = "onehot"
        self.path = path
        self.input_transform = input_transform
        self.label_transform = label_transform
        
        self.categories = categories
        print(self.path)
        print(f"feature_model: {feature_model}")
        with step_timer("loading input file paths", verbose=PROFILE):
            file_paths = get_files_of_type(self.path, ".npy", min_ctime=min_ctime)
            INPUT_TYPE = "inputs" if feature_model is None else f"{feature_model}_features"
            self.input_paths = [p for p in file_paths if INPUT_TYPE in p]
            print(f"found {len(self.input_paths)} input files for input type {INPUT_TYPE}")
        with step_timer("loading label file paths", verbose=PROFILE):
            self.label_paths = get_files_of_type(self.path, ".tsv", min_ctime=min_ctime)
            print(f"found {len(self.label_paths)} label files")
        with step_timer("creating dictionaries", verbose=PROFILE):
            self.label_dict = {os.path.basename(p).split('.')[0]: np.loadtxt(p, delimiter='\t') for p in self.label_paths}
            self.input_dict = {os.path.basename(p).split('.')[0]: p for p in self.input_paths}
            self.keys = list(self.input_dict.keys())
        print(f"Found {len(self.keys)} clips in {self.path}")
    
        label_list = []
        with step_timer("loading labels", verbose=PROFILE):
            for key in self.keys:
                video_id, frame_id = key.rsplit('_', 1)
                frame_id = int(frame_id)
                label = self.label_dict[video_id][frame_id]
                label_list.append(torch.from_numpy(label))

            # Stack into [N, C] tensor
            self.label_tensor = torch.stack(label_list).to(torch.uint8)  # or .float() if needed
            print(f"Label tensor shape: {self.label_tensor.shape}")  # [N, num_classes]

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        video_id, frame_id = key.rsplit('_', 1)
        frame_id = int(frame_id)
        with step_timer(f"loading {key}", verbose=False):
            input = torch.from_numpy(np.load(self.input_dict[key])).float()
            label = torch.from_numpy(self.label_dict[video_id][frame_id]).int() 
        if self.input_transform:
            input = self.input_transform(input)
        if self.label_transform:
            label = self.label_transform(label)
        return input, label

    def get_summary(self):
        summary = {}
        summary['metadata'] = {
            'path': self.path,
            'categories': self.categories,
            'label_type': self.label_type
        }
        label_count = self.label_tensor.sum(dim=0)
        summary['label_count'] = label_count.tolist()
        summary['dataset_size'] = len(self)
        return summary