"""
Pneumonia Detection Model Training Script
This script trains a VGG16-based model for pneumonia detection from chest X-rays
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PneumoniaDetectionTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def create_data_generators(self):
        """Create data generators for training and validation"""
        print("Creating data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=self.config['augmentation']['rescale'],
            shear_range=self.config['augmentation']['shear_range'],
            zoom_range=self.config['augmentation']['zoom_range'],
            horizontal_flip=self.config['augmentation']['horizontal_flip'],
            rotation_range=self.config['augmentation']['rotation_range'],
            width_shift_range=self.config['augmentation']['width_shift_range'],
            height_shift_range=self.config['augmentation']['height_shift_range']
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=self.config['augmentation']['rescale']
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.config['data']['train_dir'],
            target_size=(self.config['data']['img_height'], self.config['data']['img_width']),
            batch_size=self.config['data']['batch_size'],
            class_mode='categorical'
        )
        
        # Load validation data
        self.val_generator = val_datagen.flow_from_directory(
            self.config['data']['val_dir'],
            target_size=(self.config['data']['img_height'], self.config['data']['img_width']),
            batch_size=self.config['data']['batch_size'],
            class_mode='categorical'
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
        
    def build_model(self):
        """Build VGG16-based model for pneumonia detection"""
        print("Building model...")
        
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights=self.config['model']['weights'],
            include_top=self.config['model']['include_top'],
            input_shape=(self.config['data']['img_height'], 
                        self.config['data']['img_width'], 3)
        )
        
        # Freeze base model layers
        if self.config['model']['freeze_base']:
            for layer in base_model.layers:
                layer.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Add dense layers
        for units in self.config['model']['dense_units']:
            x = Dense(units, activation=self.config['model']['activation'])(x)
            x = Dropout(self.config['model']['dropout_rate'])(x)
        
        # Output layer
        num_classes = len(self.config['data']['classes'])
        predictions = Dense(num_classes, activation=self.config['model']['output_activation'])(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
            loss=self.config['training']['loss'],
            metrics=self.config['training']['metrics']
        )
        
        print("Model built successfully!")
        print(self.model.summary())
        
    def train(self):
        """Train the model"""
        print("Starting training...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            self.config['training']['checkpoint_path'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config['training']['epochs'],
            validation_data=self.val_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save final model
        final_model_path = f"models/pneumonia_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
    def plot_training_history(self):
        """Plot training history"""
        print("Plotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("Training history plot saved to: models/training_history.png")
        plt.show()

def main():
    """Main training function"""
    print("=" * 50)
    print("Pneumonia Detection Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = PneumoniaDetectionTrainer(config)
    
    # Create data generators
    trainer.create_data_generators()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    print("=" * 50)
    print("Training completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
