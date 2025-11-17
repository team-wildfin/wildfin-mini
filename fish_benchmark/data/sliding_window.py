import numpy as np
import torch
from fish_benchmark.typing.types import SlidingStyle, LocalDataset
from typing import Iterator, Callable
from dataclasses import dataclass, asdict
from math import ceil
from tqdm import tqdm
from collections import deque
from fish_benchmark.debug import step_timer
from torch.utils.data import IterableDataset, TensorDataset
from fish_benchmark.data.source import BaseSource
from fish_benchmark.data.patcher import Patcher

PROFILE = False
@dataclass
class BaseSlidingWindowDataset(IterableDataset):
    '''
    Base Class for sliding through a video and getting a window of frames. Defines sampling, patching, and shuffling.
    The returned dataset would have items of size [samples_per_window * patch_per_sample, channels, height, width]
    The total number of items is (num_frames - window_size) // step_size
    The labels are one-hot encoded.
    '''
    ds: LocalDataset
    ss: SlidingStyle
    input_transform: Callable=None, 
    total_frames: int = None
    shuffle: bool = False
    def __post_init__(self):
        self.image_window_queue = deque([], maxlen=self.ss.window_size)
        self.labels_window_queue = deque([], maxlen=self.ss.window_size)
        self.MAX_BUFFER_SIZE = 100
        self.patcher = Patcher(self.ss.patch_type, self.ss.patch_h, self.ss.patch_w)
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
        self.image_window_queue = deque([], maxlen=self.ss.window_size)
        self.labels_window_queue = deque([], maxlen=self.ss.window_size)

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
        if idx - (self.ss.window_size - 1) < 0: return False
        return (idx - (self.ss.window_size - 1)) % self.ss.step_size == 0

    def next_yielding_idx(self, idx):
        nearest_kth_yield = ceil((idx - (self.ss.window_size - 1)) / self.ss.step_size) #0 indexed 
        return nearest_kth_yield * self.ss.step_size + (self.ss.window_size - 1)

    def handle_item(self, ith_sample, image, label):
        '''
        depending on the context of the current seen images, this may return a clip or None
        '''

        if self.next_yielding_idx(ith_sample) - ith_sample > (self.ss.window_size - 1):
            #if the next yielding index is more than window size away, then this frame would not be used
            return
        
        with step_timer("converting PIL image to numpy", verbose=PROFILE): 
            #images are converted to tensors from the start as we want to treat clip processing as batch processing using torch.stack
            if not self.only_labels: self.image_window_queue.append(np.array(image.convert('RGB')))
            self.labels_window_queue.append(label)
    
        if self.is_yielding_idx(ith_sample):
            #if the calculation of yielding index is correct, then we should have enough images in the window queue
            assert len(self.labels_window_queue) >= self.ss.window_size, f"image buffer should be at least {self.ss.window_size} long"

            with step_timer("getting latest clip", verbose=PROFILE):
                if not self.only_labels: self.clips.append(self.get_latest_clip())
                self.labels.append(self.get_latest_label())

        if len(self.clips) >= self.MAX_BUFFER_SIZE:
            yield from self.flush()
            

    def get_latest_label(self):
        last_idx = len(self.labels_window_queue) - 1
        mid_idx = last_idx - int(self.ss.window_size/2)
        relevant_labels = torch.stack(list(self.labels_window_queue)[mid_idx - self.ss.tolerance_region: mid_idx + self.ss.tolerance_region + 1]) 
        relevant_labels = relevant_labels[:, :len(self.ds.categories)] #drop extra incomplete labels
        unioned_labels = torch.any(relevant_labels.bool(), dim=0)
        return unioned_labels
    
    def numpy_to_tensor(self, clip):
        '''
        np.ndarray clip has shape (samples_per_window * patch_per_sample, height, width, channels)
        '''
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).to(torch.uint8)
        return clip

    def get_latest_clip(self):
        interval = int(self.ss.window_size/self.ss.samples_per_window)
        with step_timer("stacking patches", verbose=PROFILE):
            clip = np.stack([patch 
                            for img in list(self.image_window_queue)[-self.ss.window_size::interval] 
                            for patch in self.patcher(img)])
        with step_timer("converting to tensor", verbose=PROFILE):
            tensor_clip = self.numpy_to_tensor(clip)
        with step_timer("applying vision transform", verbose=PROFILE):
            if self.input_transform:
                tensor_clip = self.input_transform(tensor_clip)

        if self.ss.data_ndim == 3: tensor_clip = tensor_clip.squeeze() # image dataset remove additional dimension
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
            yield from self.handle_item(i, image, label)

    def downsampled_length(self, video_frames_count):
        '''
        Returns give this configuration of sliding window, how many items will be generated given one video with 
        video_frames_count frames
        '''
        sampled_frames= video_frames_count
        items_count = max(0, (sampled_frames - self.ss.window_size) // self.ss.step_size)
        return items_count

    def __iter__(self):
        assert self.source is not None, "source is not set. Please set the source using set_source()"
        yield from self.scan(self.source.stream(only_labels=self.only_labels))
        yield from self.flush()

    def get_summary(self):
        summary = {}
        summary['metadata'] = asdict(self)
        label_count = torch.zeros(len(self.ds.categories))
        for label in tqdm(self.source.stream(only_labels=True)): 
            assert label.shape == (len(self.ds.categories),), f"label shape {label.shape} does not match categories {self.ds.categories}"
            label_count += label
        summary['label_count'] = label_count.tolist()
        #summary['dataset_size'] = len(self)
        return summary