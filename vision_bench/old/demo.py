'''
This module continuously classifies each frame of a video clip to a certain label. If the model used is a
vision model, then it classifies each frame individually. If the model used is a video model, then it uses a 
sliding window with length of 16 frames as input and classifies the video clip as a whole. Both scenarios take in 
a video and visulize the classification results continuously.
'''
#import abstractmethod
import cv2
from vision_bench.utils.general import read_video_pyav
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ContinuousVideoClassifier:
    '''
    Interface for continuous video classifier.
    '''
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
    
    def run(self):
        '''
        Returns to vectors, one with the frame indices and the other with the predicted labels
        '''
        indices = []
        labels = []
        device = next(self.model.parameters()).device
        print("device:", device)
        for id, batch in enumerate(self.dataloader):
            print(f"Processing batch {id}...")
            middle_idx, input = batch
            input = input.to(device)
            with torch.no_grad():
                logits = self.model(input)
            print(logits.shape)
            label = logits.argmax(dim = 1)
            indices += middle_idx.tolist()
            labels += label.tolist()
        return indices, labels

class SlidingWindowVideoDataset(Dataset):
    '''
    Dataset that generates sliding window input for video classification.
    '''
    def __init__(self, container, transform, clip_length, stride):
        self.container = container
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.indices = list(range(0, container.streams.video[0].frames - clip_length + 1, stride))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        middle_idx = start + self.clip_length // 2
        video = read_video_pyav(self.container, list(range(start, start + self.clip_length)))
        return middle_idx, self.transform(video) #shaped of video is (16, 3, 224, 224)

class RegularFramesVideoDataset(Dataset):
    '''
    Dataset that generates input for video classification.
    '''
    def __init__(self, container, transform):
        self.container = container
        self.transform = transform
        self.indices = list(range(container.streams.video[0].frames))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        video = read_video_pyav(self.container, [idx])
        transformed_single_frame_video = self.transform(video)
        frame = transformed_single_frame_video.squeeze(0) #squeeze the time dimention
        return idx, frame #shape of frame is (3, 224, 224)

def visualize(container, indices, labels, category_names, output_path="output.mp4", fps=30):
    '''
    Play the video in self.container in real time and annotate each frame (defined by indices) with the predicted label.
    Args:
        indices (`List[int]`): List of frame indices.
        labels (`List[int]`): List of predicted labels.
        category_names (`List[str]`): List of category names.
    '''
    '''
    Play the video in self.container in real-time and annotate each frame (defined by indices) with the predicted label.
    
    Args:
        indices (`List[int]`): List of frame indices.
        labels (`List[int]`): List of predicted labels.
        category_names (`List[str]`): List of category names.
    '''
    index_label_map = dict(zip(indices, labels))  # Map frame indices to labels

    # Get video stream details
    container.seek(0)
    first_frame = next(container.decode(video=0))  # Read first frame to get size
    frame_width, frame_height = first_frame.width, first_frame.height
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    cur_label = ""
    container.seek(0)
    for frame_idx, frame in enumerate(container.decode(video=0)):  # Decode video frames
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
        if frame_idx in index_label_map:
            label_idx = index_label_map[frame_idx]
            cur_label = category_names[label_idx] if label_idx < len(category_names) else str(label_idx) 
        # Draw label text on frame
        cv2.putText(
            img, cur_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        out.write(img)  # Write frame to output video

    out.release()  # Release the video writer
    print(f"Video saved to {output_path}")