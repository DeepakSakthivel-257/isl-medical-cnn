"""
Basic CNN Model architecture for ISL recognition with 5 layers
"""

import tensorflow as tf
from keras import layers, models, regularizers
from config import *

class ISLModel:
    def __init__(self, num_classes=NUM_CLASSES, input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def create_basic_cnn_model(self):
        """Create deep CNN model with 5 Conv layers for image classification"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            # 1st Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 2nd Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 3rd Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # 4th Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),

            # 5th Conv Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=LEARNING_RATE):
        """Compile the model with optimizer and loss function"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )

        return model
    
    def get_callbacks(self):
        """Get training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks

def create_model():
    """Factory function to create 5-layer CNN model"""
    isl_model = ISLModel()
    model = isl_model.create_basic_cnn_model()
    model = isl_model.compile_model(model)
    
    print("Created deep CNN model with 5 convolutional layers")
    print(f"Total parameters: {model.count_params():,}")
    
    return model, isl_model.get_callbacks()

if __name__ == "__main__":
    model, callbacks = create_model()
    model.summary()
