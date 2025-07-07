import os 
import yaml
from fish_benchmark.utils import get_files_of_type
import re
import shutil
import json

DATASET = 'coralcam'
config = yaml.safe_load(open('config/actual/dataset.yml', 'r'))
split_dict = json.load(open(f'data/organization/splits/{DATASET}.json', 'r'))
RAW_PATH = os.path.join(config[DATASET]['path'], 'raw')
DEST = os.path.join(config[DATASET]['path'], 'organized')

def get_track_id(line):
    '''
    line is of format track_<id>
    '''
    match = re.match(r'.*track_(\d+)', line)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid line format: {line}")
    
def get_video_id(line):
    '''
    Extracts the video ID before 'track_' in a string like 'GX017065_track_19219'.
    '''
    line = line.strip()
    match = re.match(r'(.+?)_track_\d+', line)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid line format: {line!r}")

if __name__ == '__main__':
    video_files = get_files_of_type(RAW_PATH, '.mp4')
    label_files = get_files_of_type(RAW_PATH, '.txt')
    label_dict = {
        os.path.splitext(os.path.basename(p))[0]: p for p in label_files
    }
    video_dict = {
        os.path.splitext(os.path.basename(p))[0]: p for p in video_files
    }
    common_keys = label_dict.keys() & video_dict.keys()
    for key in common_keys:
        video_path = video_dict[key]
        label_path = label_dict[key]
        track_id = get_track_id(key)
        video_id = get_video_id(key).upper()
        uid = f'{video_id}_{track_id}'
        split = split_dict[uid]
        dest_dir = os.path.join(DEST, split, uid)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Moving {video_path} to {os.path.join(dest_dir, uid)}")
        shutil.copy2(video_path, os.path.join(dest_dir, uid + '.mp4'))
        shutil.copy2(label_path, os.path.join(dest_dir, uid + '.txt'))