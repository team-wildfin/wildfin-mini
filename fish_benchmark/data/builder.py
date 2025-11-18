
from fish_benchmark.typing.types import LocalDataset, SlidingStyle
from fish_benchmark.data.source import SourceFactory
from fish_benchmark.data.precomputed import PrecomputedDataset
from fish_benchmark.data.sliding_window import BaseSlidingWindowDataset

class DatasetBuilder():
    def __init__(self, 
                 path: str, 
                 dataset: LocalDataset, 
                 sliding_style: SlidingStyle, 
                 input_transform=None, 
                 label_transform=None, 
                 precomputed=False, 
                 feature_model=None, 
                 min_ctime=None, 
                 only_labels=False
        ):
        self.path = path
        self.dataset = dataset
        self.sliding_style = sliding_style
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.precomputed = precomputed
        if feature_model: assert input_transform is None, "cannot transform extracted features"
        self.feature_model = feature_model
        self.min_ctime = min_ctime
        if self.min_ctime: assert precomputed, "min_ctime only works with precomputed datasets"
        self.only_labels = only_labels

    def set_transform(self, transform):
        self.input_transform = transform

    def set_only_labels(self, only_labels):
        self.only_labels = only_labels

    def build(self):
        if self.precomputed: 
            #if precomputed is true, then the sliding style information should be contained in the path
            return PrecomputedDataset(
                self.path, 
                self.dataset.categories, 
                self.input_transform, 
                self.label_transform,
                self.feature_model,
                self.min_ctime
            )
        else: 
            source = (SourceFactory.from_default(self.path, self.dataset.name)
                      .set_front_padding((self.sliding_style.window_size - 1) // 2)
                      .set_back_padding((self.sliding_style.window_size) // 2) 
                    ).build() 
            dataset = BaseSlidingWindowDataset(
                ds = self.dataset, 
                ss = self.sliding_style, 
                input_transform = self.input_transform,
                label_transform = self.label_transform,
                total_frames=source.total_frames
            )
            # if the window size is 3, then front and back padding should both be 1 so the number frames equals the number of sliding windows
            # if the window size is 4, then front padding should be 1 and back padding should be 2 so the number of frames equals the number of sliding windows
            dataset.set_source(source)
            dataset.set_only_labels(self.only_labels)
            return dataset           
