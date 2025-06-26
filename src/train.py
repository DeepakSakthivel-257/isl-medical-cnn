"""
Training script for ISL interpreter model with basic CNN
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import tensorflow as tf

from data_preprocessing import ImagePreprocessor
from model import create_model
from config import *

class ISLTrainer:
    def __init__(self):
        self.model = None
        self.callbacks = None
        self.history = None
        self.class_to_idx = None

    def load_data(self):
        """Load preprocessed data"""
        preprocessor = ImagePreprocessor()
        data = preprocessor.load_processed_data()

        if data is None:
            print("No preprocessed data found. Running preprocessing...")
            data = preprocessor.process_dataset()

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.class_to_idx = data

        print(f"Training data shape: {self.X_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Number of classes: {len(self.class_to_idx)}")

    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        return data_augmentation

    def prepare_datasets(self):
        """Prepare TensorFlow datasets"""
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))

        augment_fn = self.create_data_augmentation()
        train_dataset = train_dataset.map(
            lambda x, y: (augment_fn(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def initialize_model(self):
        """Initialize the model"""
        global NUM_CLASSES
        NUM_CLASSES = len(self.class_to_idx)

        self.model, self.callbacks = create_model()

    def train_model(self):
        """Train the model"""
        print(f"Starting training...")
        print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")

        train_dataset, val_dataset = self.prepare_datasets()

        start_time = datetime.now()

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=self.callbacks,
            verbose=1
        )

        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"Training completed in: {training_time}")

        # Save history
        with open(os.path.join('models', 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.history.history, f)

        return self.history

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history found.")
            return

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ISL Model Training History', fontsize=16, fontweight='bold')

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-3 accuracy
        if 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Train Top-3 Accuracy', linewidth=2)
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Val Top-3 Accuracy', linewidth=2)
            axes[1, 0].set_title('Top-3 Accuracy', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate or Loss Diff
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate', color='orange', linewidth=2)
            axes[1, 1].set_title('Learning Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            diff = np.array(self.history.history['val_loss']) - np.array(self.history.history['loss'])
            axes[1, 1].plot(diff, color='red', linewidth=2)
            axes[1, 1].set_title('Validation - Training Loss', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].axhline(0, linestyle='--', color='gray')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def save_model_info(self):
        """Save model metadata"""
        info = {
            'model_type': 'basic_cnn',
            'num_classes': len(self.class_to_idx),
            'class_to_idx': self.class_to_idx,
            'input_shape': self.model.input_shape,
            'total_params': self.model.count_params(),
            'training_date': datetime.now().isoformat()
        }
        with open(os.path.join('models', 'model_info.pkl'), 'wb') as f:
            pickle.dump(info, f)
        print("Model information saved.")

def main():
    print("=== ISL Interpreter Model Training ===")
    
    trainer = ISLTrainer()

    print("Loading data...")
    trainer.load_data()

    print("Initializing model...")
    trainer.initialize_model()

    trainer.model.summary()

    confirm = input("Start training model? (y/n) [y]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        print("Starting training...")
        trainer.train_model()
        trainer.plot_training_history()
        trainer.save_model_info()
        print("✅ Training completed successfully!")
    else:
        print("❌ Training cancelled.")

if __name__ == "__main__":
    main()