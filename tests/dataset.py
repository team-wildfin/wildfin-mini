from fish_benchmark.data.dataset import PrecomputedDataset
from config.data.datasets import CORALCAM, FISHFOLLOW
from fish_benchmark.data.dataset import DatasetBuilder
import os
import unittest
class TestDataset(unittest.TestCase): 
    def setUp(self): 
        original_test_cases = [
            {'dataset': CORALCAM, 'sliding_style': 'sliding_window_w_temp', 'split': 'train'},
            {'dataset': CORALCAM, 'sliding_style': 'sliding_window_w_temp', 'split': 'test'},
            {'dataset': FISHFOLLOW, 'sliding_style': 'frames_w_temp', 'split': 'train'},
            {'dataset': FISHFOLLOW, 'sliding_style': 'frames_w_temp', 'split': 'test'}
        ]
        precomputed_test_cases = [

        ]

    def test_original_dataset(self): 
        for case in self.original_test_cases: 
            try: 
                data_builder = DatasetBuilder(
                    path=os.path.join(case['dataset'].path, case['split']),
                    dataset=case['dataset'],
                    sliding_style=case['sliding_style'],
                    transform=None,
                    precomputed=False,
                    feature_model=None
                )
                data = data_builder.build()
                print(f"Dataset length: {len(data)}")
                self.assertGreater(len(data), 0, "Dataset should not be empty")

            except Exception as e:
                self.fail(f"Failed to build original dataset for case {case} with error: {e}")
            

# (benchmark) jth264@jjs533-compute-02:~/wildfin-mini$ python tests/dataset.py 
# /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# found 19803 input files for input type inputs
# found 129 label files
# Found 19803 clips in /share/j_sun/jth264/precomputed/coralcam/sliding_window_w_temp/train/
# Label tensor shape: torch.Size([19803, 5])
# Dataset length: 19803