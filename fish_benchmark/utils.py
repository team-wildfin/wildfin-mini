"""
Utility functions for data loading, video decoding, logging, and miscellaneous helpers.

Includes:
- Video frame extraction with PyAV
- Frame sampling utilities
- File and annotation helpers
- Logging setup
- Priority queue implementation
"""
import numpy as np
import torch
import os
import webdataset as wds
import json
import re
import logging
import time
from contextlib import contextmanager
import av

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    
    Example:
        container = av.open(video_path)
        frames = read_video_pyav(container, indices)
    '''
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# def load_caltech101(path, augs, train=True):
#     dataset = Caltech101(root=path, target_type = "category", transform= augs, download=True)
#     # print(dataset.categories)
#     #generate random indices to be the training set * 0.8, will split using SubSet later to be the training set 
#     random_perm = torch.randperm(len(dataset))
#     training_indices = random_perm[:int(0.8 * len(dataset))]  # 80% for training
#     testing_indices = random_perm[int(0.8 * len(dataset)):]  # 20% for testing
#     if train: 
#         res = Subset(dataset, training_indices)
#     else: 
#         res = Subset(dataset, testing_indices)
#     res.categories = dataset.categories
#     res.transform = augs
#     return res

# def load_fish_data(path, augs, train=True):
#     tar_files = [os.path.join(path, tarfile) for tarfile in os.listdir(path)]
#     dataset = wds.WebDataset(tar_files).decode("pil").to_tuple("png", "json")
#     def preprocess(sample):
#         image, json_data = sample
#         # Convert the image to a tensor and apply transformations
#         image = augs(image)
#         # Convert JSON data to a dictionary
#         json_data = json.loads(json_data)
#         # Extract the label and other information from the JSON data
#         label = json_data['label']
#         return image, label
    
#     dataset = wds.DataPipeline(
#         wds.SimpleShardList(tar_files),
#         wds.decode("pil"),
#         wds.to_tuple("png", "json"),
#     )
#     return dataset

def get_files_of_type(folder_path, file_type, min_ctime = None):
    res = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(file_type) and not file.startswith("._"):
                file_path = os.path.join(root, file)
                if min_ctime is not None: 
                    ctime = os.path.getctime(file_path)
                    if ctime < min_ctime:
                        print(f"File {file_path} has ctime {ctime} < min_ctime {min_ctime}, skipping.")
                        continue
                res.append(file_path)
    return res

def extract_annotation_identifier(filename):
    """Extracts the annotation identifier from the filename."""
    match = re.search(r'.*_([^_]*)\.csv$', filename)
    return match.group(1) if match else None

def extract_video_identifier(file_path):
    # Extract the filename (part after the last slash)
    filename = os.path.basename(file_path)
    
    # If there's an underscore, extract everything before it
    if '_' in filename:
        return filename.split('_')[0]
    
    # If there's no underscore, return the filename without extension
    return os.path.splitext(filename)[0]

import heapq

class PriorityQueue:
    def __init__(self, items=None):
        self._heap = items if items is not None else []
        heapq.heapify(self._heap)
    
    def push(self, item):
        heapq.heappush(self._heap, item)
    
    def pop(self):
        return heapq.heappop(self._heap)
    
    def peek(self):
        return self._heap[0]
    
    def is_empty(self):
        return not self._heap
    
    def size(self):
        return len(self._heap)
    
    def to_list(self):
        return sorted(self._heap)
    
def setup_logger(name, log_file = None, console = True, file = True, level=logging.INFO):
    if file is True: assert log_file is not None, "log_file must be specified if file is True"
    logger = logging.getLogger(name)
    logger.setLevel(level) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if file: 
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def frame_id_with_padding(id, padding=8):
    return str(id).zfill(padding)

def get_first_frame(path):
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(0, stream=stream, any_frame=False, backward=True)
    return next(container.decode(video=0))

def get_last_frame(path):
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(stream.duration, stream=stream, any_frame=False, backward=True)
    last_frame = None
    for frame in container.decode(video=0):
        last_frame = frame
    return last_frame