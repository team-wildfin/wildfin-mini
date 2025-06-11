import os
import re
import json
import yaml
import shutil
import subprocess
from collections import defaultdict
from fish_benchmark.utils import get_files_of_type

DATASET = 'fishfollow'
config = yaml.safe_load(open('config/actual/dataset.yml', 'r'))
split_dict = json.load(open(f'data/organization/splits/{DATASET}.json', 'r'))

RAW_PATH = os.path.join(config[DATASET]['path'], 'raw')
DEST = os.path.join(config[DATASET]['path'], 'organized')
TMP_PATH = os.path.join(config[DATASET]['path'], 'tmp_concat')
os.makedirs(TMP_PATH, exist_ok=True)

def group_video_parts(video_files):
    groups = defaultdict(list)
    for path in video_files:
        name = os.path.splitext(os.path.basename(path))[0]
        base = re.sub(r'-\d+$', '', name)
        groups[base].append(path)
    return groups

def concatenate_videos(parts, output_path):
    parts = sorted(parts)  # Sort by filename
    list_file_path = os.path.join(TMP_PATH, 'parts.txt')
    with open(list_file_path, 'w') as f:
        for part in parts:
            f.write(f"file '{os.path.abspath(part)}'\n")
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to concatenate {parts}\n{result.stderr.decode()}")

if __name__ == '__main__':
    video_files = get_files_of_type(RAW_PATH, '.mp4')
    label_files = get_files_of_type(RAW_PATH, '.txt')

    label_dict = {
        os.path.splitext(os.path.basename(p))[0]: p for p in label_files
    }

    video_groups = group_video_parts(video_files)
    for base_name, parts in video_groups.items():
        label_path = label_dict.get(base_name)
        if not label_path:
            print(f"[WARNING] No label file for base {base_name}")
            continue

        uid = base_name.upper()
        if uid not in split_dict:
            print(f"[WARNING] {uid} not in split_dict")
            continue

        split = split_dict[uid]
        dest_dir = os.path.join(DEST, split, uid)
        os.makedirs(dest_dir, exist_ok=True)

        combined_video_path = os.path.join(dest_dir, uid + '.mp4')
        concatenate_videos(parts, combined_video_path)

        shutil.copy2(label_path, os.path.join(dest_dir, uid + '.txt'))
        print(f"[INFO] Organized {uid} → {dest_dir}")
