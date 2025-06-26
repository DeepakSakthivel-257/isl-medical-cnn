"""
Data preprocessing module for converting .mov files to frames
"""

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class ImagePreprocessor:
    def __init__(self):
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.samples_per_video = FRAMES_PER_VIDEO # Number of frames to extract per video
    
    def extract_frames_from_video(self, video_path, output_dir, class_name):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // self.samples_per_video)

        frames, frame_count, extracted_count = [], 0, 0

        while cap.isOpened() and extracted_count < self.samples_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                extracted_count += 1
            frame_count += 1
        cap.release()
        
        return frames

    def process_dataset(self):
        if not os.path.exists(RAW_DATA_PATH):
            raise ValueError(f"Raw data path {RAW_DATA_PATH} does not exist")

        class_dirs = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
        if not class_dirs:
            raise ValueError("No class directories found in raw data path")
        print(f"Found {len(class_dirs)} classes: {class_dirs}")

        class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_PATH, 'class_mappings.pkl'), 'wb') as f:
            pickle.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f)

        all_images, all_labels = [], []

        for class_name in class_dirs:
            class_path = os.path.join(RAW_DATA_PATH, class_name)
            video_files = [f for f in os.listdir(class_path) 
                         if not f.startswith('.') and f.lower().endswith(('.mov', '.mp4'))]
            print(f"Processing {len(video_files)} videos for class '{class_name}'")

            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                try:
                    frames = self.extract_frames_from_video(video_path, PROCESSED_DATA_PATH, class_name)
                    if len(frames) > 0:
                        all_images.extend(frames)
                        all_labels.extend([class_to_idx[class_name]] * len(frames))
                        print(f"Processed: {video_file} - got {len(frames)} frames")
                    else:
                        print(f"Warning: No frames extracted from {video_file}")
                except Exception as e:
                    print(f"Error processing {video_file}: {str(e)}")

        if not all_images:
            raise ValueError("No images were successfully processed")

        X = np.array(all_images)
        y = np.array(all_labels)
        print(f"\nDataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {len(class_to_idx)}\n")

        if len(X) < 3:
            print("Not enough data to split — saving entire dataset as training set")
            X_train, y_train = X, y
            X_val = X_test = y_val = y_test = np.array([])
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                             test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), 
                                                             random_state=42, 
                                                             stratify=y_temp)

        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'), X_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'), y_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), X_test)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'), y_test)

        print(f"Saved:")
        print(f"   Training set:   {len(X_train)} samples")
        print(f"   Validation set: {len(X_val)} samples")
        print(f"   Test set:       {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx

    def load_processed_data(self):
        try:
            X_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'))
            y_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
            X_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'))
            y_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'))
            X_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'))
            y_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))

            with open(os.path.join(PROCESSED_DATA_PATH, 'class_mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                class_to_idx = mappings['class_to_idx']

            return X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx

        except FileNotFoundError:
            print("Processed data not found. Please run preprocessing first.")
            return None


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.process_dataset()