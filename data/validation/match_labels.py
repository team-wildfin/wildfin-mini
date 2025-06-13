import os
import av

def get_frame_count(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    frame_count = stream.frames
    container.close()
    return frame_count

def get_line_count(tsv_path):
    with open(tsv_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())

def verify_alignment(data_root):
    mismatches = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
        for shard in os.listdir(split_dir):
            shard_path = os.path.join(split_dir, shard)
            if not os.path.isdir(shard_path):
                continue
            for file in os.listdir(shard_path):
                if not file.endswith('.mp4'):
                    continue
                base = os.path.splitext(file)[0]
                video_path = os.path.join(shard_path, base + '.mp4')
                tsv_path = os.path.join(shard_path, base + '.txt')
                if not os.path.exists(tsv_path):
                    print(f"[WARNING] TSV missing for {video_path}")
                    continue
                try:
                    frames = get_frame_count(video_path)
                    lines = get_line_count(tsv_path)
                    if frames != lines:
                        mismatches.append((video_path, frames, lines))
                        print(f"[ERROR] Mismatch in {video_path}: {frames} frames vs {lines} lines")
                except Exception as e:
                    print(f"[ERROR] Failed to process {video_path}: {e}")
    print(f"[INFO] Done. {len(mismatches)} mismatches found.")
    return mismatches

if __name__ == "__main__":
    DATA_ROOT = "/share/j_sun/jth264/fishfollow/organized"  # <-- Change this
    verify_alignment(DATA_ROOT)
