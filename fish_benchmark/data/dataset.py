import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, TensorDataset, Sampler
import os
import av
from fish_benchmark.utils import read_video_pyav, sample_frame_indices, get_first_frame, get_last_frame
from torchvision.datasets import Caltech101
from torch.utils.data import Subset
import webdataset as wds
import json
from abc import ABC, abstractmethod
from fish_benchmark.utils import get_files_of_type
import random
import yaml
from typing import Iterator, Callable, Optional
from dataclasses import dataclass, asdict
from math import ceil
from tqdm import tqdm
from collections import deque
from fish_benchmark.debug import step_timer
from torchvision.transforms import ToTensor
from pydantic import BaseModel, ConfigDict 
from typing import Callable, Optional, List 
import logging
import math 
import yaml

to_tensor = ToTensor()
PROFILE = False
PROFILE_LOADING = True
logger = logging.getLogger(__name__)
dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))

def onehot(num_total_classes, active_classes):
    """
    Convert a list of class indices to one-hot encoding.
    """
    one_hot = torch.zeros(num_total_classes, dtype=torch.float32)
    one_hot[active_classes] = 1
    return one_hot

def parse_annotation(annotation):
    behaviors = []
    for event in annotation['events']:
        behaviors.append(event['behavior']['name'])
    return behaviors

def get_sample_indices(lst_len, clip_len, sample_method = "even-spaced"):
    if sample_method == "even-spaced":
        indices = np.linspace(0, lst_len - 1, clip_len).astype(int)
    elif sample_method == "random":
        indices = np.random.choice(lst_len, clip_len, replace=False)
    else:
        raise ValueError(f"sample_method {sample_method} not recognized.")
    return indices

class BaseSource:
    def __iter__(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def stream(self, only_labels: bool, front_padding: int, back_padding: int) -> Iterator: 
        raise NotImplementedError("Subclasses should implement this method")

class BaseConfig: 
    def get_config(self):
        required_attrs = ['categories', 'label_type']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute '{attr}' in {self.__class__.__name__}")
        return {
            'categories': self.categories,
            'label_type': self.label_type,
        }
    
class Patcher: 
    def __init__(self, patch_type, h, w):
        assert patch_type in ["absolute", "relative"], f"patch_type {patch_type} not recognized"
        # if patch_type == "absolute": h is the height of each patch, w is the width of each patch
        # if patch_type == "relative": the full height of the image would be cut into h patches, and the full width of the image would be cut into w patches
        self.patch_type = patch_type
        self.h = h
        self.w = w
    
    def pad_to_multiple_np(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Pads a HWC (Height-Width-Channel) image so that its height is a multiple of h
        and width is a multiple of w using 0-padding.

        Args:
            image (np.ndarray): Input image of shape (H, W, C)
            h (int): Desired height multiple
            w (int): Desired width multiple

        Returns:
            np.ndarray: Padded image
        """
        H, W, C = image.shape
        pad_h = (h - H % h) % h
        pad_w = (w - W % w) % w
        padding = ((0, pad_h), (0, pad_w), (0, 0))  # Pad bottom and right
        return np.pad(image, padding, mode='constant', constant_values=0)

    def patch(self, img, h, w):
        img = self.pad_to_multiple_np(img, h, w)
        return img.reshape(-1, h, w, img.shape[2])
    
    def __call__(self, img):
        if self.patch_type == "absolute":
            return self.patch(img, self.h, self.w)
        elif self.patch_type == "relative":
            # Split the image into patches
            h = math.ceil(img.shape[0] / self.h) #h is the 
            w = math.ceil(img.shape[1] / self.w)
            return self.patch(img, h, w)

@dataclass
class BaseSlidingWindowDataset(IterableDataset):
    '''
    Base Class for sliding through a video and getting a window of frames. Defines sampling, patching, and shuffling.
    The returned dataset would have items of size [samples_per_window * patch_per_sample, channels, height, width]
    The total number of items is (num_frames - window_size) // step_size
    The labels are one-hot encoded.
    '''
    input_transform: Callable=None, 
    label_type: str= "onehot", 
    window_size: int=16,  
    tolerance_region: int = 16, 
    samples_per_window: int= 16, 
    step_size: int = 1, 
    categories: list = None, 
    data_ndim: int = None, 
    shuffle: bool = False,
    patch_type: str = 'relative', 
    patch_h: int = 1, 
    patch_w: int = 1,
    temporal_sample_interval: int = 1,
    MAX_BUFFER_SIZE: int = 100
    total_frames: int = None
    def __post_init__(self):
        assert self.window_size % self.samples_per_window == 0, f"window_size {self.window_size} should be a factor of samples_per_window {self.samples_per_window}"
        assert self.temporal_sample_interval > 0, f"temporal_sample_interval {self.temporal_sample_interval} should be greater than 0"
        assert self.tolerance_region <= (self.window_size - 1)//2, f"tolerance_region {self.tolerance_region} should be less than or equal to window_size {self.window_size//2}"
        assert self.label_type == "onehot", 'currently only onehot is supported'
        if self.data_ndim == 3: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.labels_window_queue = deque([], maxlen=self.window_size)
        self.patcher = Patcher(self.patch_type, self.patch_h, self.patch_w)
        self.only_labels = False
        self.clips = []
        self.labels = []
        self.source = None
    def set_only_labels(self, only_labels: bool):
        self.only_labels = only_labels

    def set_source(self, source: BaseSource):
        '''
        Source iterator should yield (image, label) tuples with images being PIL images and label 
        being a pytorch tensor of shape (num_classes,)
        '''
        self.source = source

    def __len__(self):
        return self.downsampled_length(self.total_frames) if self.total_frames else None

    def clear_window_queue(self):
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.labels_window_queue = deque([], maxlen=self.window_size)

    def clear_buffer(self):
        self.clips = []
        self.labels = []

    def flush(self):
        if len(self.labels) == 0: 
            self.clear_buffer()
            return
        
        if not self.only_labels: 
            clips = torch.stack(self.clips)
            labels = torch.stack(self.labels)
            if self.shuffle: 
                perm = torch.randperm(len(clips))
                clips = clips[perm]
                labels = labels[perm]
            dataset = TensorDataset(clips, labels)
            for image, label in dataset:
                yield image, label
        else: 
            labels = torch.stack(self.labels)
            if self.shuffle:
                perm = torch.randperm(len(labels))
                labels = labels[perm]
            for label in labels: 
                yield (None, label)
                
        self.clear_buffer()

    def is_yielding_idx(self, idx):
        if idx - (self.window_size - 1) < 0: return False
        return (idx - (self.window_size - 1)) % self.step_size == 0
    
    def next_yielding_idx(self, idx):
        nearest_kth_yield = ceil((idx - (self.window_size - 1)) / self.step_size) #0 indexed 
        return nearest_kth_yield * self.step_size + (self.window_size - 1)

    def handle_item(self, ith_sample, image, label):
        '''
        depending on the context of the current seen images, this may return a clip or None
        '''
        
        if self.next_yielding_idx(ith_sample) - ith_sample > (self.window_size - 1):
            #if the next yielding index is more than window size away, then this frame would not be used
            return
        
        with step_timer("converting PIL image to numpy", verbose=PROFILE): 
            #images are converted to tensors from the start as we want to treat clip processing as batch processing using torch.stack
            if not self.only_labels: self.image_window_queue.append(np.array(image.convert('RGB')))
            self.labels_window_queue.append(label)
    
        if self.is_yielding_idx(ith_sample):
            #if the calculation of yielding index is correct, then we should have enough images in the window queue
            assert len(self.labels_window_queue) >= self.window_size, f"image buffer should be at least {self.window_size} long"

            with step_timer("getting latest clip", verbose=PROFILE):
                if not self.only_labels: self.clips.append(self.get_latest_clip())
                self.labels.append(self.get_latest_label())

        if len(self.clips) >= self.MAX_BUFFER_SIZE:
            yield from self.flush()
            

    def get_latest_label(self):
        last_idx = len(self.labels_window_queue) - 1
        mid_idx = last_idx - int(self.window_size/2)
        relevant_labels = torch.stack(list(self.labels_window_queue)[mid_idx - self.tolerance_region: mid_idx + self.tolerance_region + 1]) 
        relevant_labels = relevant_labels[:, :len(self.categories)] #drop extra incomplete labels
        unioned_labels = torch.any(relevant_labels.bool(), dim=0)
        return unioned_labels
    
    def numpy_to_tensor(self, clip):
        '''
        np.ndarray clip has shape (samples_per_window * patch_per_sample, height, width, channels)
        '''
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).to(torch.uint8)
        return clip

    def get_latest_clip(self):
        interval = int(self.window_size/self.samples_per_window)
        with step_timer("stacking patches", verbose=PROFILE):
            clip = np.stack([patch 
                            for img in list(self.image_window_queue)[-self.window_size::interval] 
                            for patch in self.patcher(img)])
        with step_timer("converting to tensor", verbose=PROFILE):
            tensor_clip = self.numpy_to_tensor(clip)
        with step_timer("applying vision transform", verbose=PROFILE):
            if self.input_transform:
                tensor_clip = self.input_transform(tensor_clip)

        if self.data_ndim == 3: tensor_clip = tensor_clip.squeeze() # image dataset remove additional dimension
        return tensor_clip

    def scan(self, annotated_video_frames: Iterator):
        '''
        annotated_video_frames is a generator that yields (image, annotation) tuples
        image is a PIL image
        label is whatever the dataset label. One needs to implement the annotation_to_label function to convert it to a label
        '''
        self.clear_buffer()
        self.clear_window_queue()
        for i, (image, label) in enumerate(annotated_video_frames):
            if i % self.temporal_sample_interval == 0: 
                sample_idx = i // self.temporal_sample_interval
                yield from self.handle_item(sample_idx, image, label)

    def downsampled_length(self, video_frames_count):
        '''
        Returns give this configuration of sliding window, how many items will be generated given one video with 
        video_frames_count frames
        '''
        sampled_frames= video_frames_count // self.temporal_sample_interval
        items_count = max(0, (sampled_frames - self.window_size) // self.step_size)
        return items_count

    def __iter__(self):
        assert self.source is not None, "source is not set. Please set the source using set_source()"
        yield from self.scan(self.source.stream(only_labels=self.only_labels))
        yield from self.flush()

    def get_summary(self):
        summary = {}
        summary['metadata'] = asdict(self)
        label_count = torch.zeros(len(self.categories))
        for label in tqdm(self.source.stream(only_labels=True)): 
            assert label.shape == (len(self.categories),), f"label shape {label.shape} does not match categories {self.categories}"
            label_count += label
        summary['label_count'] = label_count.tolist()
        #summary['dataset_size'] = len(self)
        return summary

def get_categories(dataset_name):
    if dataset_name in dataset_config:
        return dataset_config[dataset_name]['categories']
    else:
        raise ValueError(f"dataset_name {dataset_name} not recognized")

class FrameAnnotatedSource(BaseSource):
    '''
    Data is stored in a folder with the following structure:
    /path/to/data/
        ├── shard1/
            |videoid1.mp4
            |videoid1.tsv
            |videoid2.mp4
            |videoid2.tsv
        ├── shard2/
        │
    Each row of the tsv files is the annotation for the respective frame in the video
    Requires: 
    1. video in the mp4 has the same number of frames as the rows of the tsv file
    2. vidoes and labels have the same name, except for the extension
    '''
    def __init__(self, path, video_file_type, label_file_type, front_padding = 0, back_padding = 0):
        '''
        video_file_type example: ".mp4"
        label_file_type example: ".tsv"
        make sure the txt file is tab separated. 
        front_padding and back_padding are the number of frames to pad at the beginning and end of the video. 
        This is useful when the sliding window using this source wants to sample frames from the beginning and end of the video.
        '''
        self.path = path
        self.label_type = "onehot"
        self.front_padding = front_padding
        self.back_padding = back_padding
        self.annotation_to_label = lambda x: torch.tensor(x)
        self.video_paths = get_files_of_type(self.path, video_file_type)
        self.label_paths = get_files_of_type(self.path, label_file_type)
        self.label_dict = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in self.label_paths
        }
        self.total_frames = self.calculate_total_frames()

    def calculate_total_frames(self):
        total_frames = 0
        for track_path in self.video_paths:
            track_name = os.path.splitext(os.path.basename(track_path))[0]
            label_path = self.label_dict.get(track_name)
            if label_path is None:
                continue
            
            container = av.open(track_path)
            total_frames += container.streams.video[0].frames
            total_frames += self.front_padding + self.back_padding
        return total_frames
    
    def stream(self, only_labels = False):
        for video_path in self.video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label_path = self.label_dict.get(video_name)
            if label_path is None:
                continue

            label = np.loadtxt(label_path, delimiter='\t', dtype=int)
            container = av.open(video_path)
            assert label.shape[0] == container.streams.video[0].frames, f"Number of frames in {video_path} does not match number of annotations in {label_path}"

            #front padding
            first_frame = get_first_frame(video_path)
            for i in range(self.front_padding):
                if only_labels:
                    yield None, self.annotation_to_label(label[0])
                else: 
                    yield first_frame.to_image(), self.annotation_to_label(label[0])
            
            #actual frames
            video_frames = container.decode(video=0)
            for i in range(label.shape[0]):
                if only_labels:
                    yield None, self.annotation_to_label(label[i])
                else: 
                    frame = next(video_frames)
                    yield frame.to_image(), self.annotation_to_label(label[i])

            #back padding
            last_frame = get_last_frame(video_path)
            for i in range(self.back_padding):
                if only_labels:
                    yield None, self.annotation_to_label(label[-1])
                else: 
                    yield last_frame.to_image(), self.annotation_to_label(label[-1])

def get_dicts_and_common_keys(list1, list2):
    '''
    lists are file paths and keys are file names
    '''
    dict1 = {os.path.basename(p).split('.')[0]: p for p in list1}
    dict2 = {os.path.basename(p).split('.')[0]: p for p in list2}
    common_keys = list(dict1.keys() & dict2.keys())
    return dict1, dict2, common_keys

class VideoAnnotatedSource(BaseSource):
    '''
    Data is stored in a folder with the following structure:
    /path/to/data/
        ├── class1/
            |videoid1.avi
            |videoid1.txt
    txt contains a single number representing the class index
    '''
    def __init__(self, path, n_classes, front_padding = 0, back_padding = 0):
        self.path = path
        self.label_type = "onehot"  
        self.annotation_to_label=lambda x: onehot(len(n_classes), [x])
        self.video_paths = sorted(get_files_of_type(self.path, '.avi'))
        self.annotation_paths = sorted(get_files_of_type(self.path, ".txt"))
        self.video_dict, self.annotation_dict, self.keys = get_dicts_and_common_keys(self.video_paths, self.annotation_paths)
        self.total_frames = self.calculate_total_frames()
        self.front_padding = front_padding
        self.back_padding = back_padding

    def calculate_total_frames(self):
        total_frames = 0
        for key in self.keys:
            video_path = self.video_dict[key]
            container = av.open(video_path)
            total_frames += container.streams.video[0].frames
        return total_frames

    def stream(self, only_labels = False):
        for key in self.keys:
            video_path = self.video_dict[key]
            annotation_path = self.annotation_dict[key]
            with open(annotation_path, 'r') as f:
                annotation_idx = int(f.read().strip())

            #front padding
            first_frame = get_first_frame(video_path)
            for i in range(self.front_padding):
                if only_labels:
                    yield None, self.annotation_to_label(annotation_idx)
                else: 
                    yield first_frame.to_image(), self.annotation_to_label(annotation_idx)

            #actual frames
            container = av.open(video_path)
            video_frames = container.decode(video=0)
            total_frames = container.streams.video[0].frames
            for i in range(total_frames):
                if only_labels:
                    yield None, self.annotation_to_label(annotation_idx)
                else: 
                    frame = next(video_frames)
                    yield frame.to_image(), self.annotation_to_label(annotation_idx)

            #back padding
            last_frame = get_last_frame(video_path)
            for i in range(self.back_padding):
                if only_labels:
                    yield None, self.annotation_to_label(annotation_idx)
                else: 
                    yield last_frame.to_image(), self.annotation_to_label(annotation_idx)

class SourceFactory: 
    def __init__(self, path, source_type, video_file_type, label_file_type, front_padding = 0, back_padding = 0, n_classes = None):
        self.path = path
        self.source_type = source_type
        self.front_padding = front_padding
        self.back_padding = back_padding
        self.n_classes = n_classes
        self.video_file_type = video_file_type
        self.label_file_type = label_file_type
        self.n_classes = n_classes

    def set_front_padding(self, front_padding):
        self.front_padding = front_padding
        return self

    def set_back_padding(self, back_padding):
        self.back_padding = back_padding
        return self
    
    def build(self):
        if self.source_type == 'video_annotated':
            assert self.n_classes is not None, "n_classes should be set for video_annotated source"
            return VideoAnnotatedSource(self.path, self.n_classes, self.front_padding, self.back_padding)
        elif self.source_type == 'frame_annotated':
            return FrameAnnotatedSource(
                self.path, 
                self.video_file_type, 
                self.label_file_type, 
                self.front_padding, 
                self.back_padding
            )
        
    @classmethod
    def from_default(cls, path, dataset_name):
        if dataset_name == 'ucf101':
            return cls(
                path=path,
                source_type='video_annotated',
                n_classes=len(get_categories('ucf101'))
            )
        elif dataset_name == 'fishfollow':
            return cls(
                path=path,
                source_type='frame_annotated',
                video_file_type='.mp4',
                label_file_type='.tsv'
            )
        elif dataset_name == 'coralcam':
            return cls(
                path=path,
                source_type='frame_annotated',
                video_file_type='.mp4',
                label_file_type='.txt'
            )
        else:
            raise ValueError(f"dataset_name {dataset_name} not recognized")

class PrecomputedDataset(Dataset):
    '''
    Dataset mounted on precomputed sliding window clips and labels. 
    Corresponding clips and labels have the same name but live in different folders
    '''
    def __init__(self, path, categories, transform=None, feature_model=None, min_ctime=None):
        '''
        path should be contain 2 subfolders: frames and labels
        '''
        self.label_type = "onehot"
        self.path = path
        self.transform = transform
        self.categories = categories
        print(self.path)
        with step_timer("loading input file paths", verbose=PROFILE):
            file_paths = get_files_of_type(self.path, ".npy", min_ctime=min_ctime)
            INPUT_TYPE = "inputs" if feature_model is None else f"{feature_model}_features"
            self.input_paths = [p for p in file_paths if INPUT_TYPE in p]
            print(f"found {len(self.input_paths)} input files")
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
        if self.transform:
            input = self.transform(input)
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

class SlidingWindowConfig(BaseModel):
    window_size: int
    tolerance_region: int 
    samples_per_window: int
    step_size: int
    data_ndim: int  
    shuffle:  bool
    patch_type: str
    patch_h: int
    patch_w: int
    temporal_sample_interval: int
    MAX_BUFFER_SIZE: int

class DatasetConfig(BaseModel):
    categories: list
    label_type: str
    model_config = ConfigDict(extra="ignore")

class DatasetBuilder():
    def __init__(self, path, dataset_name, style, transform=None, precomputed=False, feature_model=None, min_ctime=None, only_labels=False):
        self.path = path
        self.dataset_name = dataset_name
        self.set_sliding_window_config(yaml.safe_load(open("config/sliding_style.yml", "r"))[style])
        self.set_dataset_config(dataset_config[dataset_name])
        self.input_transform = None
        self.transform = transform
        self.precomputed = precomputed
        if feature_model: assert transform is None, "cannot transform extracted features"
        self.feature_model = feature_model
        self.min_ctime = min_ctime
        if self.min_ctime: assert precomputed, "min_ctime only works with precomputed datasets"
        self.only_labels = only_labels

    def set_sliding_window_config(self, config_dict):
        self.swconfig = SlidingWindowConfig(**config_dict)

    def set_transform(self, transform):
        self.transform = transform

    def set_only_labels(self, only_labels):
        self.only_labels = only_labels

    def set_dataset_config(self, config_dict):
        self.dsconfig = DatasetConfig(**config_dict)

    def set_dsconfig_attr(self, attr_name, value):
        if not hasattr(self.dsconfig, attr_name):
            raise AttributeError(f"Config has no attribute '{attr_name}'")
        setattr(self.dsconfig, attr_name, value)

    def set_swconfig_attr(self, attr_name, value):
        if not hasattr(self.swconfig, attr_name):
            raise AttributeError(f"Config has no attribute '{attr_name}'")
        setattr(self.swconfig, attr_name, value)

    def build(self):
        if self.precomputed: 
            #if precomputed is true, then the sliding style information should be contained in the path
            return PrecomputedDataset(
                self.path, 
                self.dsconfig.categories, 
                self.transform, 
                self.feature_model,
                self.min_ctime
            )
        else: 
            source = (SourceFactory.from_default(self.path, self.dataset_name)
                      .set_front_padding((self.swconfig.window_size - 1) // 2)
                      .set_back_padding((self.swconfig.window_size) // 2) 
                    ).build() 
            config = {
                **self.swconfig.model_dump(), 
                **self.dsconfig.model_dump(), 
                'total_frames': source.total_frames, 
                'input_transform': self.transform,
            }
            dataset = BaseSlidingWindowDataset(**config)
            # if the window size is 3, then front and back padding should both be 1 so the number frames equals the number of sliding windows
            # if the window size is 4, then front padding should be 1 and back padding should be 2 so the number of frames equals the number of sliding windows
            dataset.set_source(source)
            dataset.set_only_labels(self.only_labels)
            return dataset   
        

class MultiLabelBalancedSampler(Sampler):
    def __init__(self, dataset, max_samples_per_class=1000):
        """
        Args:
            dataset: A dataset with a `.label_tensor` attribute of shape [N, C] (multi-hot labels).
            max_samples_per_class (int): Maximum number of samples to draw for each class per epoch.
        """
        if not hasattr(dataset, "label_tensor"):
            raise ValueError("Dataset must have a `.label_tensor` attribute of shape [N, num_classes]")
        
        self.label_tensor = dataset.label_tensor
        #augment the label tensor with a dummy class for the "no label" case
        self.label_tensor = torch.cat([self.label_tensor, (self.label_tensor.sum(dim=1, keepdim=True) == 0).float()], dim=1)
        print(f"label_tensor shape: {self.label_tensor.shape}")
        self.num_classes = self.label_tensor.shape[1]
        print(f"self.num_classes: {self.num_classes}")
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
            if not self.non_zero[class_id]: continue
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