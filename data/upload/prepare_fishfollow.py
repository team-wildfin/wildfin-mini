import os
import av
import shutil
import yaml
import subprocess
from config.datasets import DATASETS

# --- Constants ---
DATASET_NAME = 'fishfollow'
DATASET = DATASETS[DATASET_NAME]
FRAME_THRESHOLD = 5000  # max frames per part

ORG_ROOT = os.path.join(DATASET.path)
OUTPUT_DIR = os.path.join('/share/j_sun/jth264/fishfollow', 'prepared_uploads')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_video_pyav(input_path, output_prefix, frame_threshold=5000):
    container = av.open(input_path)
    video_stream = container.streams.video[0]
    audio_streams = [s for s in container.streams if s.type == "audio"]

    part_idx = 0
    frame_count = 0
    total_frame_count = 0
    part_frame_counts = []

    out_container = None
    out_video_stream = None
    out_audio_stream = None

    def open_new_output(index):
        out_path = f"{output_prefix}-{index}.mp4"
        oc = av.open(out_path, mode='w')
        vs = oc.add_stream(template=video_stream)
        as_ = oc.add_stream(template=audio_streams[0]) if audio_streams else None
        return oc, vs, as_, out_path

    out_container, out_video_stream, out_audio_stream, _ = open_new_output(part_idx)
    part_frame_counts.append(0)

    for packet in container.demux():
        for frame in packet.decode():
            if packet.stream.type == "video":
                if frame_count >= frame_threshold:
                    # Flush and close current
                    for p in out_video_stream.encode():
                        out_container.mux(p)
                    if out_audio_stream:
                        for p in out_audio_stream.encode():
                            out_container.mux(p)
                    out_container.close()

                    # Start new part
                    part_idx += 1
                    frame_count = 0
                    out_container, out_video_stream, out_audio_stream, _ = open_new_output(part_idx)
                    part_frame_counts.append(0)

                packet = out_video_stream.encode(frame)
                if packet:
                    out_container.mux(packet)

                frame_count += 1
                total_frame_count += 1
                part_frame_counts[-1] += 1

            elif packet.stream.type == "audio" and out_audio_stream:
                packet.pts = None  # ensure clean timing
                out_container.mux(packet)

    # Final flush
    for p in out_video_stream.encode():
        out_container.mux(p)
    if out_audio_stream:
        for p in out_audio_stream.encode():
            out_container.mux(p)
    out_container.close()
    container.close()

    # ✅ Verification
    summed = sum(part_frame_counts)
    print(f"[INFO] Total frames: {total_frame_count}")
    print(f"[INFO] Sum of part frames: {summed}")
    assert total_frame_count == summed, f"❌ Frame mismatch! total={total_frame_count} vs sum={summed}"
    print(f"[✅] Verified: all {total_frame_count} frames accounted for across {len(part_frame_counts)} parts")


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
                split_video_pyav(input_path, output_prefix, frame_threshold=5000)
            elif file.endswith('.txt'):
                uid = os.path.splitext(file)[0]
                input_path = os.path.join(root, file)
                output_path = os.path.join(OUTPUT_DIR, f"{uid}.txt")
                shutil.copy2(input_path, output_path)
                print(f"[INFO] Copied label {input_path}")


if __name__ == '__main__':
    prepare_upload_files()
    print(f"[✅] All files prepared in {OUTPUT_DIR}")