import os
import re
import json
import yaml
import shutil
import subprocess
from collections import defaultdict
from fish_benchmark.utils.general import get_files_of_type
import av  # PyAV for video processing
from config.data.datasets import DATASETS

DATASET_NAME = 'fishfollow'
DATASET = DATASETS[DATASET_NAME]
split_dict = json.load(open(f'data/organization/splits/{DATASET_NAME}.json', 'r'))

RAW_PATH = os.path.join(os.path.dirname(DATASET.path), 'raw')
DEST = os.path.join(DATASET.path)
TMP_PATH = os.path.join(DATASET.path, 'tmp_concat')
os.makedirs(TMP_PATH, exist_ok=True)

def group_video_parts(video_files):
    groups = defaultdict(list)
    for path in video_files:
        name = os.path.splitext(os.path.basename(path))[0]
        base = re.sub(r'-\d+$', '', name)
        groups[base].append(path)

    for base in groups:
        if len(groups[base]) > 1: groups[base].sort(key=lambda p: int(re.search(r'-(\d+)', os.path.splitext(os.path.basename(p))[0]).group(1)))

    return groups

def total_frames(parts):
    total = 0
    for part in parts:
        try:
            container = av.open(part)
            stream = container.streams.video[0]
            total += stream.frames
            container.close()
        except Exception as e:
            raise RuntimeError(f"PyAV failed to count frames for {part}: {e}")
    return total

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

def validate(video_groups, label_dict): 
    print("[INFO] Validating video groups and labels")
    for base_name, parts in video_groups.items():
        label_path = label_dict.get(base_name)
        if not label_path:
            print(f"[WARNING] No label file for base {base_name}")
            continue
        uid = base_name.upper()
        if uid not in split_dict:
            print(f"[WARNING] {uid} not in split_dict")
            continue
        with open(label_path, 'r') as f:
            label_lines = [line for line in f if line.strip()]
        assert total_frames(parts) == len(label_lines), f"Frame count mismatch for {uid}: {total_frames(parts)} video frames vs {len(label_lines)} labels"
        
def organize_files(video_groups, label_dict): 
    for base_name, parts in video_groups.items():
        print(f"[INFO] Processing base {base_name} with {len(parts)} parts")
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

if __name__ == '__main__':
    video_files = get_files_of_type(RAW_PATH, '.mp4')
    label_files = get_files_of_type(RAW_PATH, '.txt')
    print(f"[INFO] Found {len(video_files)} video files and {len(label_files)} label files in {RAW_PATH}")
    label_dict = {
        os.path.splitext(os.path.basename(p))[0]: p for p in label_files
    }
    
    video_groups = group_video_parts(video_files)
    print(f"[INFO] Found {len(video_groups)} video groups with parts")
    validate(video_groups, label_dict)
    organize_files(video_groups, label_dict)
    print(f"[INFO] Files organized successfully in {DEST}")