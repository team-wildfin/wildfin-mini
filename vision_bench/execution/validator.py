from typing import Dict, Type
import os
import yaml
from vision_bench.typing.types import LocalDataset, Split, SlidingStyle
import av
import logging
from config.maps.model_sliding_style import MODEL_SLIDING_STYLES
from config.maps.sliding_style_test import TEST_NAME
from config.data.sliding_styles import SLIDING_STYLES
from vision_bench.management.manager import ShardManager


class Validator: 
    def __init__(self, 
                 root_path: str,
                 shard_manager: Type[ShardManager] = ShardManager,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.root_path = root_path
        self.shard_manager = shard_manager

    def find_report(self, dataset, split, sliding_style, model) -> Dict: 
        # report should exist in REPORT_ROOT/dataset/split/sliding_style/<model>_report.yml
        report_path = os.path.join(
            self.root_path, dataset, split, sliding_style, f"{model}_report.yml"
        )
        if not os.path.exists(report_path):
            self.logger.warning(
                f"Report not found for {dataset} {split} {sliding_style} {model} at {report_path}"
            )
            return {'train': {}, 'val': {}, 'test': {}}  # empty report
        with open(report_path, "r") as f:
            report = yaml.safe_load(f)
        return report
    
    @staticmethod
    def get_complete_subsets(report: Dict):
        # assert report is not empty
        assert report, "Report is empty"
        data: Dict = list(report.values())[
            0
        ]  # Assuming report is a dict with one key-value pair
        res = []
        for subset, subset_data in data.items():
            if subset_data["actual_files"] == subset_data["expected_items"]:
                res.append(subset)
        return res

    @staticmethod 
    def calculate_expected_files(video_path, style: SlidingStyle):
        '''
        calculate the expected number of files produced by applying sliding_style to video in video_path
        '''
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        padded_frames = total_frames + (style.window_size // 2) + ((style.window_size - 1) // 2) 
        # sampled_frames = (padded_frames - 1) // style['temporal_sample_interval'] + 1
        sampled_frames = padded_frames  # assuming temporal_sample_interval is 1
        expected_files = max(0, (sampled_frames - style.window_size) // style.step_size + 1)
        return expected_files
    
    @staticmethod
    def validate_features(expected_files, output_path):
        '''
        check if output path contains the expected number of files produced by applying sliding_style to video in video_path
        '''
        if not os.path.exists(output_path):
            return False, expected_files, 0
        #.npy files
        actual_files = len([f for f in os.listdir(output_path) if f.endswith('.npy')])
        is_valid = actual_files == expected_files
        return is_valid, expected_files, actual_files

    @staticmethod
    def validate_labels(expected_files, label_path):
        '''
        check of the label file has the expected number of rows produced by applying sliding_style to video in video_path
        Requires: label path to be a .tsv or .txt file. Return 0 if the file does not exist or is empty
        '''
        if not os.path.exists(label_path):
            return False, expected_files, 0
        with open(label_path, 'r') as f:
            actual_lines = sum(1 for line in f if line.strip())
        is_valid = actual_lines == expected_files
        return is_valid, expected_files, actual_lines

    def compute(self, dataset: LocalDataset, split: Split, sliding_style: SlidingStyle, feature_extractor: str) -> Dict:
        report = {}
        shard_manager = self.shard_manager(dataset.precomputed_path)
        split_path = os.path.join(dataset.path, split.name)
        report[split.name] = {}
        for subset in os.listdir(split_path): 
            self.logger.debug(f"Validating {dataset.name} {sliding_style.name} {feature_extractor} for split {split.name}, subset {subset}")
            subset_path = os.path.join(split_path, subset) 
            video_path = os.path.join(subset_path, f'{subset}.mp4')
            feature_type_name = feature_extractor if feature_extractor == 'inputs' else f'{feature_extractor}_features'
            output_path = shard_manager.locate_shard(split=split.name, subset=subset, sliding_style=sliding_style.name, shard_type=feature_type_name)
            label_path = os.path.join(
                shard_manager.locate_shard(split=split.name, subset=subset, sliding_style=sliding_style.name, shard_type='labels'), 
                f"{subset}.tsv"
            )
            expected_items = Validator.calculate_expected_files(video_path, sliding_style)
            feature_good, _, actual_files = Validator.validate_features(expected_items, output_path)
            label_good, _, actual_lines = Validator.validate_labels(expected_items, label_path)
            report[split.name][subset] = {
                'valid': feature_good and label_good,
                'expected_items': expected_items,
                'actual_files': actual_files,
                'label_lines': actual_lines,
            }
        return report
    

    def run(self, dataset: LocalDataset, sliding_style: SlidingStyle, model: str): 
        '''
        model can be 'inputs' or feature extractor name
        sliding_style should be name of training sliding style. The testing sliding style will be determined based on mapping.  
        '''
        self.logger.info(f"Running validator for {dataset.name} with sliding style {sliding_style.name} and model {model}")
        for split in dataset.splits:   
            ss = (sliding_style if sliding_style in split.sliding_styles else 
                 (SLIDING_STYLES[TEST_NAME[sliding_style.name]]
                  if SLIDING_STYLES[TEST_NAME[sliding_style.name]] in split.sliding_styles 
                  else None))
            if not ss: 
                self.logger.warning(f"No valid sliding style found for {dataset.name} {split.name} with requested style {sliding_style.name}. Skipping validation for this split.")
                continue
           
            if (model != "inputs" and 
                ss.name not in [s.name for s in MODEL_SLIDING_STYLES[model]]): continue
            report = self.compute(dataset, split, ss, model)
            report_path = os.path.join(self.root_path, dataset.name, split.name, ss.name, f'{model}_report.yml')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                yaml.dump(report, f)
            self.logger.info(f"Validation report saved to {report_path}")
            self.logger.debug(f"Report: {report}")