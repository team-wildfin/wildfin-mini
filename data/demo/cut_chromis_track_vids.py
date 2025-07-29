import cv2
import numpy as np
import os
import pickle

def process_video(video_path, bbox_file_path, extract_dim, n_pixels_crop, output_dir):
    """
    Process a video by extracting tracked objects and creating separate videos for each tracked object.

    Args:
        video_path (str): The path to the input video file.
        bbox_file_path (str): The path to the file containing bounding box data.
        extract_dim (tuple): The dimensions (width, height) of the extracted region for each tracked object.
        n_pixels_crop (int): The number of pixels to crop from each side of the extracted region.
        output_dir (str): The directory where the output videos will be saved.

    Returns:
        None
    """
    # Load all_bboxes
    with open(bbox_file_path, 'rb') as f:
        all_bboxes = pickle.load(f)

    # def create_bbox_presence_table(all_bboxes):
    #     # Find the total number of frames and the unique track IDs
    #     total_frames = len(all_bboxes)
    #     unique_track_ids = set()
    #     for frame_data in all_bboxes:
    #         if frame_data:  # Check if frame_data is not empty
    #             for bbox_info in frame_data:
    #                 unique_track_ids.add(bbox_info["track_id"])

    #     # Sort the track IDs for consistent ordering
    #     unique_track_ids = sorted(list(unique_track_ids))

    #     # Create a dictionary to hold bbox presence data
    #     bbox_presence = {track_id: ['No' for _ in range(total_frames)] for track_id in unique_track_ids}

    #     # Populate the dictionary with bbox presence data
    #     for frame_idx, frame_data in enumerate(all_bboxes):
    #         if frame_data:  # Check if frame_data is not empty
    #             for bbox_info in frame_data:
    #                 track_id = bbox_info["track_id"]
    #                 bbox_presence[track_id][frame_idx] = 'Yes'

    #     return bbox_presence




    def interpolate_bbox(bbox1, bbox2, steps_before, steps_after):
        if bbox1 is None or bbox2 is None:
            return None
        if steps_before + steps_after == 0:
            return bbox1
        alpha = steps_before / (steps_before + steps_after)
        interpolated = [int(b1 + alpha * (b2 - b1)) for b1, b2 in zip(bbox1, bbox2)]
        return tuple(interpolated)

    def fill_gaps_and_interpolate(all_bboxes):
        # Find the unique track IDs
        unique_track_ids = set()
        for frame_data in all_bboxes:
            for bbox_info in frame_data:
                unique_track_ids.add(bbox_info["track_id"])

        # Process each track ID
        for track_id in unique_track_ids:
            last_known_bbox = None
            last_known_frame_idx = -1
            for frame_idx in range(len(all_bboxes)):
                current_bbox = None
                for bbox_info in all_bboxes[frame_idx]:
                    if bbox_info["track_id"] == track_id:
                        current_bbox = bbox_info["bbox"]
                        break

                if current_bbox is not None:
                    if last_known_bbox is not None and frame_idx > last_known_frame_idx + 1:
                        # There is a gap to fill
                        for gap_idx in range(last_known_frame_idx + 1, frame_idx):
                            steps_before = gap_idx - last_known_frame_idx
                            steps_after = frame_idx - gap_idx
                            interpolated_bbox = interpolate_bbox(last_known_bbox, current_bbox, steps_before, steps_after)
                            all_bboxes[gap_idx].append({"track_id": track_id, "bbox": interpolated_bbox})

                    last_known_bbox = current_bbox
                    last_known_frame_idx = frame_idx

    def create_tracked_videos(video_path, all_bboxes, extract_dim, n_pixels_crop, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        track_videos = {}
        last_appearance = {}  # Dictionary to track the last appearance of each track_id

        # Determine the last appearance of each track_id
        for frame_idx, frame_bboxes in enumerate(all_bboxes):
            print(frame_idx)
            for bbox_info in frame_bboxes:
                track_id = bbox_info["track_id"]
                last_appearance[track_id] = frame_idx

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print(frame_idx)
            h, w, _ = frame.shape
            if frame_idx < len(all_bboxes):
                frame_bboxes = all_bboxes[frame_idx]
            
            for bbox_info in frame_bboxes:
                if bbox_info["bbox"] is not None:
                    x1, y1, x2, y2 = map(int, bbox_info["bbox"])
                    track_id = int(bbox_info["track_id"])

                    # Calculate width and height of the bounding box
                    bb_width = x2 - x1
                    bb_height = y2 - y1
                    x1 = float(x1)
                    x2 = float(x2)
                    y1 = float(y1)
                    y2 = float(y2)
                    if bb_width > bb_height:
                        new_y2 = (y2 + y1)/ 2 + bb_width/ 2
                        y1 = (y2 + y1)/ 2 - bb_width/ 2
                        y2 = new_y2
                    else:
                        new_x2 = (x2 + x1)/ 2 + bb_height/ 2
                        x1 = (x2 + x1)/ 2 - bb_height/ 2
                        x2 = new_x2
                    bb_width = x2 - x1
                    bb_height = y2 - y1

                    if bb_width != bb_height:
                        print(bb_width, bb_height)

                    enlargement_factor = 0.3
                    x1 = max(int(x1 - enlargement_factor * bb_width), 0)
                    x2 = min(int(x2 + enlargement_factor * bb_width), w)
                    y1 = max(int(y1 - enlargement_factor * bb_height), 0)
                    y2 = min(int(y2 + enlargement_factor * bb_height), h)

                    bb_width = x2 - x1
                    bb_height = y2 - y1

                    if bb_width != bb_height:
                        print(bb_width, bb_height)

                    # Determine the region to extract
                    if bb_width > extract_dim[0] or bb_height > extract_dim[1]:
                        # Resize the bounding box
                        scale_factor = min(extract_dim[0] / bb_width, extract_dim[1] / bb_height)
                        resized_width = int(bb_width * scale_factor)
                        resized_height = int(bb_height * scale_factor)
                        resized_region = cv2.resize(frame[y1:y2, x1:x2], (resized_width, resized_height), interpolation=cv2.INTER_AREA)

                        # Create a new image and place the resized region in the center
                        extracted_region = np.zeros((extract_dim[1], extract_dim[0], 3), dtype=np.uint8)
                        start_x = (extract_dim[0] - resized_width) // 2
                        start_y = (extract_dim[1] - resized_height) // 2
                        extracted_region[start_y:start_y+resized_height, start_x:start_x+resized_width] = resized_region
                    else:
                        # Extract region centered on BB's midpoint
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        start_x = max(center_x - extract_dim[0] // 2, 0)
                        start_y = max(center_y - extract_dim[1] // 2, 0)
                        end_x = min(start_x + extract_dim[0], w)
                        end_y = min(start_y + extract_dim[1], h)
                        extracted_region = frame[start_y:end_y, start_x:end_x]

                    # Now, crop 'n_pixels_crop' from each side of the extracted region
                    if extracted_region.shape[0] > 2 * n_pixels_crop and extracted_region.shape[1] > 2 * n_pixels_crop:
                        extracted_region = extracted_region[n_pixels_crop:-n_pixels_crop, n_pixels_crop:-n_pixels_crop]

                    # Resize back to the extract_dim (if needed)
                    if extracted_region.shape[:2] != (extract_dim[1], extract_dim[0]):
                        extracted_region = cv2.resize(extracted_region, (extract_dim[0], extract_dim[1]), interpolation=cv2.INTER_AREA)

                    # Add text information (bb dimensions and frame number)
                    bb_area = bb_width * bb_height
                    text_info = f"A: {bb_area}, F: {frame_idx}"
                    cv2.putText(extracted_region, text_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    text_info = f"x: {((x1 + x2) / 2 / w):.2f}, y: {((y1 + y2) / 2 / h):.2f}"
                    cv2.putText(extracted_region, text_info, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 1, cv2.LINE_AA)

                    if track_id not in track_videos:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_path = os.path.join(output_dir, f"track_{track_id}.mp4")
                        track_videos[track_id] = cv2.VideoWriter(output_path, fourcc, 30.0, extract_dim)

                    track_videos[track_id].write(extracted_region)

            for track_id, last_idx in last_appearance.items():
                if frame_idx > last_idx and track_id in track_videos:
                    track_videos[track_id].release()
                    del track_videos[track_id]

            frame_idx += 1

        cap.release()
        for track_id, writer in track_videos.items():
            print(f"Track {track_id}: Number of frames written = {writer.get(cv2.CAP_PROP_FRAME_COUNT)}")
            writer.release()
    # Call the necessary functions

    with open(bbox_file_path, 'rb') as f:
        all_bboxes = pickle.load(f)
    #create_bbox_presence_table(all_bboxes)
    fill_gaps_and_interpolate(all_bboxes)
    create_tracked_videos(video_path, all_bboxes, extract_dim, n_pixels_crop, output_dir)
