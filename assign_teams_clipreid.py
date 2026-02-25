import argparse
import os
import glob
import numpy as np
import torch
import cv2
from tqdm import tqdm
from loguru import logger
from typing import List
from team_classifier_clipreid import TeamClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
SHRINK_SCALE = 1.0

# def shrink_boxes(xyxy: np.ndarray, scale: float) -> np.ndarray:
#     """
#     Shrinks bounding boxes by a given scale factor while keeping their centers fixed.
#     This helps in focusing on the jersey color by removing background noise.

#     Args:
#         xyxy (np.ndarray): Array of shape (N, 4) in [x1, y1, x2, y2] format.
#         scale (float): Scale factor (e.g., 0.4).

#     Returns:
#         np.ndarray: Resized bounding boxes.
#     """
#     x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
#     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#     w, h = (x2 - x1) * scale, (y2 - y1) * scale

#     new_x1, new_y1 = cx - w / 2, cy - h / 2
#     new_x2, new_y2 = cx + w / 2, cy + h / 2

#     return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)

def get_crops_from_frame(img: np.ndarray, detections: np.ndarray, shrink_scale: float = 0.4) -> List[np.ndarray]:
    """
    Crops player images from the frame based on detections (MOT format).

    Args:
        img (np.ndarray): The full video frame.
        detections (np.ndarray): Detections array for this frame.
        shrink_scale (float): Factor to shrink the box for better color sampling.

    Returns:
        List[np.ndarray]: A list of cropped images (numpy arrays).
    """
    xyxy = detections[:, 2:6].copy()
    xyxy[:, 2] += xyxy[:, 0]
    xyxy[:, 3] += xyxy[:, 1]

    # shrunk_xyxy = shrink_boxes(xyxy, scale=shrink_scale)
    
    crops = []
    h_img, w_img = img.shape[:2]
    
    for box in xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        crops.append(img[y1:y2, x1:x2])
            
    return crops

def main(data_path, pred_dir, output_dir):
    """
    Main execution loop:
    1. Sampling: Collect crops from a subset of frames.
    2. Fitting: Train the TeamClassifier on sampled crops.
    3. Inference: Predict team IDs for ALL frames.
    4. Saving: Write new MOT files with Team ID appended.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    team_classifier = TeamClassifier(device=DEVICE, batch_size=BATCH_SIZE)

    seqs = sorted([file for file in os.listdir(pred_dir) if file.endswith('.txt')])

    for s_id, seq_file in enumerate(seqs, 1):
        seq_name = seq_file.replace('.txt', '')
        logger.info(f"Processing Sequence: {seq_name} ({s_id}/{len(seqs)})")
        
        track_path = os.path.join(pred_dir, seq_file)
        track_res = np.genfromtxt(track_path, dtype=float, delimiter=',')
        
        img_dir = os.path.join(data_path, seq_name, 'img1')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*')))
        
        if len(track_res) == 0:
            logger.warning(f"Empty tracking file: {seq_file}")
            continue

        # ==========================================
        # Phase 1: Data Collection (Sampling & Fitting)
        # ==========================================
        logger.info("Phase 1: Sampling crops for training...")
        training_crops = []
        
        unique_frames = np.unique(track_res[:, 0]).astype(int)
        
        # Sampling Stride: Sample 1 frame every 30 frames
        STRIDE = 30
        sampled_frames = unique_frames[::STRIDE]
        
        for frame_id in tqdm(sampled_frames, desc="Collecting Samples"):
            if frame_id - 1 < len(img_files):
                img_path = img_files[frame_id - 1]
                img = cv2.imread(img_path)
                
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                frame_dets = track_res[track_res[:, 0] == frame_id]
                crops = get_crops_from_frame(img, frame_dets, shrink_scale=SHRINK_SCALE)
                training_crops.extend(crops)

        logger.info(f"Fitting TeamClassifier with {len(training_crops)} samples...")
        if len(training_crops) > 0:
            team_classifier.fit(training_crops)
        else:
            logger.error("No training crops collected! Skipping sequence.")
            continue

        # ==========================================
        # Phase 2: Global Inference (Inference on All Frames)
        # ==========================================
        logger.info("Phase 2: Predicting teams for all frames...")
        
        final_output_data = []

        for frame_id in tqdm(unique_frames, desc="Inference"):
            if frame_id - 1 >= len(img_files): 
                continue 
                
            img_path = img_files[frame_id - 1]
            img = cv2.imread(img_path)
            
            if img is None: continue

            frame_dets = track_res[track_res[:, 0] == frame_id]
            if len(frame_dets) == 0:
                continue

            # 1. Get Crops
            crops = get_crops_from_frame(img, frame_dets, shrink_scale=SHRINK_SCALE)
            
            # 2. Predict Team ID (0 or 1)
            team_ids = team_classifier.predict(crops)
            
            # 3. Combine Results: Append Team ID to the end of the row
            for det_row, t_id in zip(frame_dets, team_ids):
                new_row = np.append(det_row, t_id)
                final_output_data.append(new_row)

        # ==========================================
        # 3. Save Results
        # ==========================================
        output_filepath = os.path.join(output_dir, seq_file)
        if final_output_data:
            final_array = np.stack(final_output_data)
            np.savetxt(
                output_filepath, 
                final_array,
                fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d',
                delimiter=','
            )
            logger.info(f"Saved results with Team IDs to: {output_filepath}")
        else:
            logger.warning(f"No results generated for {seq_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign Team IDs using SigLIP + KMeans.")
    parser.add_argument('--data_path', type=str, required=True, help="Root directory containing image sequences.")
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing MOT txt files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save new MOT txt files.")
    
    args = parser.parse_args()
    
    main(args.data_path, args.pred_dir, args.output_dir)