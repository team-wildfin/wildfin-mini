from fish_benchmark.data.dataset import PrecomputedDataset
from config.datasets import CORALCAM
if __name__ == '__main__':

    dataset = PrecomputedDataset(
        path = '/share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/', 
        categories=CORALCAM.categories,
        transform = None, 
        feature_model = None,
    ) 
    print(f"Dataset length: {len(dataset)}")
# (benchmark) jth264@jjs533-compute-02:~/wildfin-mini$ python tests/dataset.py 
# /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# found 19803 input files for input type inputs
# found 129 label files
# Found 19803 clips in /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# Label tensor shape: torch.Size([19803, 5])
# Dataset length: 19803