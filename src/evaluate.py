"""
Model evaluation and metrics generation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pickle
import os
from datetime import datetime
from collections import Counter

from data_preprocessing import ImagePreprocessor
from config import *

class ISLEvaluator:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model_path = model_path
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
        preprocessor = ImagePreprocessor()
        data = preprocessor.load_processed_data()
        
        if data is None:
            raise ValueError("No preprocessed data found. Run preprocessing first.")
        
        _, _, self.X_test, _, _, self.y_test, self.class_to_idx = data
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Number of test samples: {len(self.y_test)}")
        
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        predictions = self.model.predict(self.X_test, batch_size=BATCH_SIZE, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        top_3_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(
            self.y_test, predictions, k=3
        ).numpy().mean()
        
        top_5_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(
            self.y_test, predictions, k=5
        ).numpy().mean()
        
        print("\n=== Model Evaluation Results ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top_3_accuracy': top_3_accuracy,
            'top_5_accuracy': top_5_accuracy,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_PATH, 'evaluation_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        return y_pred, predictions, metrics
    
    def plot_confusion_matrix(self, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        class_names = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - ISL Sign Recognition', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
                transform=plt.gca().transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_per_class_metrics(self, y_pred):
        """Plot per-class performance metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred, average=None
        )
        
        class_names = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        
        axes[0, 0].bar(class_names, precision, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(class_names, recall, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(class_names, f1, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(class_names, support, color='lightyellow', alpha=0.7)
        axes[1, 1].set_title('Support per Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'per_class_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_confidence(self, predictions):
        """Plot prediction confidence distribution"""
        max_confidences = np.max(predictions, axis=1)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(max_confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Prediction Confidence')
        plt.xlabel('Max Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        correct_predictions = (np.argmax(predictions, axis=1) == self.y_test)
        correct_conf = max_confidences[correct_predictions]
        incorrect_conf = max_confidences[~correct_predictions]
        
        plt.subplot(2, 2, 2)
        plt.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
        plt.title('Confidence Distribution by Correctness')
        plt.ylabel('Confidence Score')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_confidences >= confidence_bins[i]) & (max_confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct_predictions[mask])
                bin_accuracies.append(bin_acc)
            else:
                bin_accuracies.append(0)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        plt.title('Reliability Diagram')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        k_values = range(1, min(6, len(self.class_to_idx) + 1))
        top_k_accs = []
        
        for k in k_values:
            top_k_acc = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                self.y_test, predictions, k=k
            ).numpy().mean()
            top_k_accs.append(top_k_acc)
        
        plt.plot(k_values, top_k_accs, 'o-', linewidth=2, markersize=8)
        plt.title('Top-K Accuracy')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'prediction_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_pred):
        """Generate and save detailed classification report"""
        class_names = [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
        
        report = classification_report(
            self.y_test, y_pred, 
            target_names=class_names,
            digits=4
        )
        
        print("\n=== Detailed Classification Report ===")
        print(report)
        
        with open(os.path.join(RESULTS_PATH, 'classification_report.txt'), 'w') as f:
            f.write("ISL Sign Recognition - Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report)
        
        return report
    
    def analyze_misclassifications(self, y_pred, predictions, top_n=10):
        """Analyze most common misclassifications"""
        misclassified_mask = (y_pred != self.y_test)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        misclass_details = []
        for idx in misclassified_indices:
            true_class = self.idx_to_class[self.y_test[idx]]
            pred_class = self.idx_to_class[y_pred[idx]]
            confidence = predictions[idx][y_pred[idx]]
            
            misclass_details.append({
                'index': idx,
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence
            })
        
        misclass_details.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n=== Top {top_n} Misclassifications (High Confidence Errors) ===")
        for i, detail in enumerate(misclass_details[:top_n]):
            print(f"{i+1}. Sample {detail['index']}: "
                  f"True: {detail['true_class']}, "
                  f"Predicted: {detail['pred_class']}, "
                  f"Confidence: {detail['confidence']:.4f}")
        
        with open(os.path.join(RESULTS_PATH, 'misclassification_analysis.txt'), 'w') as f:
            f.write("ISL Sign Recognition - Misclassification Analysis\n")
            f.write("=" * 55 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total misclassifications: {len(misclassified_indices)}\n")
            f.write(f"Misclassification rate: {len(misclassified_indices)/len(self.y_test):.4f}\n\n")
            
            f.write("Top Misclassifications (High Confidence Errors):\n")
            f.write("-" * 50 + "\n")
            for i, detail in enumerate(misclass_details[:20]):
                f.write(f"{i+1}. Sample {detail['index']}: "
                       f"True: {detail['true_class']}, "
                       f"Predicted: {detail['pred_class']}, "
                       f"Confidence: {detail['confidence']:.4f}\n")

def main():
    """Main evaluation function"""
    print("=== ISL Interpreter Model Evaluation ===")
    
    evaluator = ISLEvaluator()
    
    try:
        print("Loading model and test data...")
        evaluator.load_model_and_data()
        
        y_pred, predictions, metrics = evaluator.evaluate_model()
        
        print("Generating confusion matrix...")
        evaluator.plot_confusion_matrix(y_pred)
        
        print("Generating per-class metrics...")
        evaluator.plot_per_class_metrics(y_pred)
        
        print("Analyzing prediction confidence...")
        evaluator.plot_prediction_confidence(predictions)
        
        print("Generating classification report...")
        evaluator.generate_classification_report(y_pred)
        
        print("Analyzing misclassifications...")
        evaluator.analyze_misclassifications(y_pred, predictions)
        
        print(f"\nEvaluation completed! Results saved in '{RESULTS_PATH}' directory.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()