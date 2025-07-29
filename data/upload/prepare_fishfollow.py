import os
import av
import shutil
import yaml
import subprocess

# --- Constants ---
DATASET = 'fishfollow'
FRAME_THRESHOLD = 5000  # max frames per part
config = yaml.safe_load(open('config/actual/dataset.yml', 'r'))

ORG_ROOT = os.path.join(config[DATASET]['path'])
OUTPUT_DIR = os.path.join('/share/j_sun/jth264/fishfollow', 'prepared_uploads')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def count_frames(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    frames = sum(1 for _ in container.decode(stream))
    container.close()
    return frames


def split_video_ffmpeg(input_path, output_prefix, segment_seconds=60):
    """
    Split video by time and verify frame counts across split parts.
    """
    output_template = f"{output_prefix}-%d.mp4"

    # Step 1: Count frames in the original video
    print(f"[INFO] Counting frames in original: {input_path}")
    original_frame_count = count_frames(input_path)
    print(f"[INFO] Original frame count: {original_frame_count}")

    # Step 2: Run FFmpeg to split
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c", "copy",
        "-map", "0:v", "-map", "0:a?",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        output_template
    ]
    print(f"[INFO] Running FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode()}")

    # Step 3: Count total frames across split parts
    base_dir = os.path.dirname(output_prefix)
    base_name = os.path.basename(output_prefix)
    part_files = sorted(
        f for f in os.listdir(base_dir)
        if f.startswith(base_name + "-") and f.endswith(".mp4")
    )
    total_split_frames = 0
    for part in part_files:
        part_path = os.path.join(base_dir, part)
        frames = count_frames(part_path)
        print(f"[INFO] {part} → {frames} frames")
        total_split_frames += frames

    print(f"[INFO] Total frames across parts: {total_split_frames}")
    assert total_split_frames == original_frame_count, f"❌ Frame count mismatch! {total_split_frames} vs {original_frame_count}"
    print(f"[✅] Frame counts match.")

def prepare_upload_files():
    # --- Clear OUTPUT_DIR before starting ---
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Clearing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Walk and process files ---
    for root, dirs, files in os.walk(ORG_ROOT):
        for file in files:
            if file.endswith('.mp4'):
                uid = os.path.splitext(file)[0]
                input_path = os.path.join(root, file)
                output_prefix = os.path.join(OUTPUT_DIR, uid)
                print(f"[INFO] Splitting {input_path}")
                split_video_ffmpeg(input_path, output_prefix, segment_seconds=60)
            elif file.endswith('.txt'):
                uid = os.path.splitext(file)[0]
                input_path = os.path.join(root, file)
                output_path = os.path.join(OUTPUT_DIR, f"{uid}.txt")
                shutil.copy2(input_path, output_path)
                print(f"[INFO] Copied label {input_path}")


if __name__ == '__main__':
    prepare_upload_files()
    print(f"[✅] All files prepared in {OUTPUT_DIR}")