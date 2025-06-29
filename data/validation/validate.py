import av
import yaml
import os
from fish_benchmark.utils import setup_logger
config = yaml.safe_load(open("config/sliding_style.yml", "r"))
dataset_config = yaml.safe_load(open("config/actual/dataset.yml", "r"))
model_config =yaml.safe_load(open("config/models.yml", "r"))
DATASETS = [
    # 'coralcam', 
    'fishfollow'
]
FEATURE_EXTRACTORS = ['dino', 'dino_large', 'videomae']  # or 'clip', etc.
SAVE_DIR = 'data/validation/reports'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
logger = setup_logger(
    "validate_sliding_style",
    console=True,
    file=False,
)
def calculate_expected_files(video_path, sliding_style):
    '''
    calculate the expected number of files produced by applying sliding_style to video in video_path
    '''
    container = av.open(video_path)
    style = config[sliding_style]
    total_frames = container.streams.video[0].frames
    padded_frames = total_frames + (style['window_size'] // 2) + ((style['window_size'] - 1) // 2) 
    sampled_frames = (padded_frames - 1) // style['temporal_sample_interval'] + 1
    expected_files = max(0, (sampled_frames - style['window_size']) // style['step_size'] + 1)
    return expected_files

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

def run(dataset, split, sliding_style, feature_extractor):
    report = {}
    dset = dataset_config[dataset]
    split_path = os.path.join(dset['path'], split)
    report[split] = {}
    for subset in os.listdir(split_path): 
        logger.info(f"Validating {dataset} {sliding_style} {feature_extractor} for split {split}, subset {subset}")
        subset_path = os.path.join(split_path, subset) 
        video_path = os.path.join(subset_path, f'{subset}.mp4')
        output_path = os.path.join(dset['precomputed_path'], sliding_style, split, subset, f'{feature_extractor}_features')
        label_path = os.path.join(dset['precomputed_path'], sliding_style, split, subset, 'labels', f'{subset}.tsv')
        expected_items = calculate_expected_files(video_path, sliding_style)
        feature_good, _, actual_files = validate_features(expected_items, output_path)
        label_good, _, actual_lines = validate_labels(expected_items, label_path)
        report[split][subset] = {
            'valid': feature_good and label_good,
            'expected_items': expected_items,
            'actual_files': actual_files,
            'label_lines': actual_lines,
        }
    return report

if __name__ == '__main__':
    for DATASET in DATASETS: 
        for SPLIT in dataset_config[DATASET]['splits'].keys():   
            for SLIDING_STYLE in dataset_config[DATASET]['splits'][SPLIT]['sliding_styles']: 
                for FEATURE_EXTRACTOR in FEATURE_EXTRACTORS:
                    if model_config[FEATURE_EXTRACTOR]['sliding_styles'] and SLIDING_STYLE not in model_config[FEATURE_EXTRACTOR]['sliding_styles']: continue
                    report = run(DATASET, SPLIT, SLIDING_STYLE, FEATURE_EXTRACTOR)
                    report_path = os.path.join(SAVE_DIR, DATASET, SPLIT, SLIDING_STYLE, f'{FEATURE_EXTRACTOR}_report.yml')
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, 'w') as f:
                        yaml.dump(report, f)
                    print(f"Validation report saved to {report_path}")
                    print("Report:", report)