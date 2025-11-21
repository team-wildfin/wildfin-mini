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
import json
import re
import logging
import time
from contextlib import contextmanager
import av
from typing import Any, Callable, List, Union
import glob

def get_files_of_type(folder_path, file_type):
    res = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(file_type) and not file.startswith("._"):
                file_path = os.path.join(root, file)
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

def find_single_match(pattern): 
    files = glob.glob(pattern)
    assert len(files) == 1, f"Expected exactly one checkpoint file matching pattern in {pattern}"
    file = files[0]