"""
Model Evaluation Script
Evaluates the trained pneumonia detection model on test data
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class ModelEvaluator:
    def __init__(self, model_path, config):
        self.config = config
        self.model = load_model(model_path)
        self.test_generator = None
        
    def create_test_generator(self):
        """Create test data generator"""
        print("Creating test data generator...")
        
        test_datagen = ImageDataGenerator(
            rescale=self.config['augmentation']['rescale']
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.config['data']['test_dir'],
            target_size=(self.config['data']['img_height'], self.config['data']['img_width']),
            batch_size=self.config['data']['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.test_generator.class_indices}")
        
    def evaluate(self):
        """Evaluate model on test data"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = list(self.test_generator.class_indices.keys())
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        return accuracy, y_pred, y_true
        
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        print("\nConfusion matrix saved to: models/confusion_matrix.png")
        plt.show()
        
    def plot_sample_predictions(self, num_samples=9):
        """Plot sample predictions"""
        print("\nGenerating sample predictions...")
        
        # Reset generator
        self.test_generator.reset()
        
        # Get a batch of images
        images, labels = next(self.test_generator)
        predictions = self.model.predict(images[:num_samples])
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        class_names = list(self.test_generator.class_indices.keys())
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            true_label = class_names[np.argmax(labels[i])]
            pred_label = class_names[np.argmax(predictions[i])]
            confidence = np.max(predictions[i]) * 100
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/sample_predictions.png')
        print("Sample predictions saved to: models/sample_predictions.png")
        plt.show()

def main():
    """Main evaluation function"""
    print("=" * 50)
    print("Pneumonia Detection Model Evaluation")
    print("=" * 50)
    
    # Model path
    model_path = config['deployment']['model_path']
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, config)
    
    # Create test generator
    evaluator.create_test_generator()
    
    # Evaluate model
    accuracy, y_pred, y_true = evaluator.evaluate()
    
    # Plot sample predictions
    evaluator.plot_sample_predictions()
    
    print("\n" + "=" * 50)
    print("Evaluation completed successfully!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
