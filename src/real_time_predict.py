"""
Real-time ISL recognition using webcam with basic CNN
"""

import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from collections import deque, Counter
import time
from datetime import datetime
import sys

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

class RealTimeISLPredictor:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model_path = model_path
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.prediction_history = deque(maxlen=10)
        self.last_prediction = ""
        self.last_confidence = 0.0

        # Initialize webcam
        self.cap = None
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

    def load_model_and_mappings(self):
        """Load trained model and class mappings"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print("Loading model...")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")

        mappings_path = os.path.join(PROCESSED_DATA_PATH, 'class_mappings.pkl')
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Class mappings not found at {mappings_path}")

        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.class_to_idx = mappings['class_to_idx']
            self.idx_to_class = mappings['idx_to_class']

        print(f"Loaded {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")

    def initialize_camera(self, camera_index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"Camera {camera_index} initialized")

    def preprocess_frame(self, frame):
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        return frame_normalized

    def predict_sign(self, frame):
        processed_frame = self.preprocess_frame(frame)
        frame_expanded = np.expand_dims(processed_frame, axis=0)
        predictions = self.model.predict(frame_expanded, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.idx_to_class[predicted_class_idx]
        return predicted_class, confidence

    def smooth_predictions(self, prediction, confidence):
        if confidence > PREDICTION_THRESHOLD:
            self.prediction_history.append(prediction)

        if len(self.prediction_history) >= 3:
            most_common = Counter(self.prediction_history).most_common(1)
            if most_common:
                return most_common[0][0]

        return prediction if confidence > PREDICTION_THRESHOLD else ""

    def draw_ui(self, frame, prediction_text, confidence):
        display_text = f"Prediction: {prediction_text} ({confidence*100:.1f}%)"
        fps_text = f"FPS: {self.current_fps:.2f}"

        cv2.putText(frame, display_text, (10, 30), self.font, self.font_scale, (0, 255, 0), self.thickness)
        cv2.putText(frame, fps_text, (10, 60), self.font, self.font_scale, (255, 255, 0), self.thickness)

        return frame

    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        if elapsed_time >= 1.0:
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time

    def save_prediction(self, prediction, confidence):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"prediction_{now}.txt"
        with open(filename, 'w') as f:
            f.write(f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
        print(f"Saved prediction to {filename}")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")

    def run_real_time_prediction(self):
        print("Starting real-time ISL recognition...")
        print("Press 'q' to quit, 'r' to reset history, 's' to save prediction")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                prediction, confidence = self.predict_sign(frame)

                if prediction:
                    smoothed_prediction = self.smooth_predictions(prediction, confidence)
                    if smoothed_prediction:
                        self.last_prediction = smoothed_prediction
                        self.last_confidence = confidence

                display_frame = self.draw_ui(
                    frame,
                    self.last_prediction if self.last_confidence > PREDICTION_THRESHOLD else prediction,
                    confidence
                )

                cv2.imshow('ISL Real-time Recognition', display_frame)
                self.update_fps()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.prediction_history.clear()
                    self.last_prediction = ""
                    self.last_confidence = 0.0
                    print("Prediction history reset")
                elif key == ord('s'):
                    if self.last_prediction and self.last_confidence > PREDICTION_THRESHOLD:
                        self.save_prediction(self.last_prediction, self.last_confidence)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        except Exception as e:
            print(f"Error during prediction: {str(e)}")

        finally:
            self.cleanup()


def main():
    print("=== ISL Real-time Recognition ===")

    try:
        predictor = RealTimeISLPredictor()
        predictor.load_model_and_mappings()

        camera_index = input(f"Enter camera index (default: {CAMERA_INDEX}): ").strip()
        if camera_index.isdigit():
            camera_index = int(camera_index)
        else:
            camera_index = CAMERA_INDEX

        predictor.initialize_camera(camera_index)
        predictor.run_real_time_prediction()

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have:")
        print("1. Trained model available")
        print("2. Webcam connected")
        print("3. All required dependencies installed")


if __name__ == "__main__":
    main()
