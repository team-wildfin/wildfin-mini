from fish_benchmark.utils.general import get_files_of_type, get_first_frame, get_last_frame
from typing import Iterator
import os 
import numpy as np
import torch
import av
import logging
logger = logging.getLogger(__name__)

class BaseSource:
    def __iter__(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def stream(self, only_labels: bool, front_padding: int, back_padding: int) -> Iterator: 
        raise NotImplementedError("Subclasses should implement this method")

class FrameAnnotatedSource(BaseSource):
    '''
    Data is stored in a folder with the following structure:
    /path/to/data/
        |-- train
            ├── shard1/
                |videoid1.mp4
                |videoid1.tsv
            ├── shard2/
                |videoid2.mp4
                |videoid2.tsv
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
            if label.shape[0] != container.streams.video[0].frames:
                logger.warning(f"Label length {label.shape[0]} does not match number of frames {container.streams.video[0].frames} in video {video_path}")
                continue

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
            return VideoAnnotatedSource(
                self.path, 
                self.n_classes, 
                self.front_padding, 
                self.back_padding
            )
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
        
        if dataset_name == 'fishfollow':
            return cls(
                path=path,
                source_type='frame_annotated',
                video_file_type='.mp4',
                label_file_type='.txt'
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
