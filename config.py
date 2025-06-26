"""
Configuration file for ISL Interpreter
"""

import os

MODEL_SAVE_PATH = 'models/isl_cnn_model.h5'
PROCESSED_DATA_PATH = 'features'
CAMERA_INDEX = 0
FRAME_WIDTH = 64
FRAME_HEIGHT = 64
PREDICTION_THRESHOLD = 0.7

# Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/best_model.h5"
RESULTS_PATH = "results"

# Data preprocessing
FRAME_WIDTH = 244
FRAME_HEIGHT = 244
FRAMES_PER_VIDEO = 30  # Number of frames to extract per video
TARGET_FPS = 10  # Target FPS for frame extraction

# Model parameters
NUM_CLASSES = 26  # Adjust based on your dataset (A-Z for alphabet)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Real-time prediction
CAMERA_INDEX = 0
PREDICTION_THRESHOLD = 0.7
FRAME_BUFFER_SIZE = 10

# Create directories
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)