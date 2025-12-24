"""Utility functions for model evaluation and result management."""

import csv
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels from model
    
    Returns:
        Tuple of (accuracy, precision, recall, f1_score)
        
    Example:
        >>> metrics = evaluate_model(y_test, y_pred)
        >>> accuracy, precision, recall, f1 = metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def save_results(model_name, ngram_range, metrics, training_time=None, results_csv="data/results.csv"):
    """
    Save model evaluation results to CSV file (auto-creates file if needed).
    
    Creates or appends to results CSV with columns:
    [Model, Ngram_Range, Accuracy, Precision, Recall, F1, Training_Time]
    
    Args:
        model_name: Name of the model (e.g., 'SVM', 'Logistic Regression')
        ngram_range: Tuple of n-gram range, e.g., (1, 2) for unigrams + bigrams
        metrics: Tuple of (accuracy, precision, recall, f1) scores
        training_time: Training time in seconds (optional, defaults to empty)
        results_csv: Path to results CSV file (default: 'data/results.csv')
    
    Example:
        >>> metrics = evaluate_model(y_test, y_pred)
        >>> save_results('SVM', (1,2), metrics, training_time=45.2)
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    
    # Create CSV with headers if it doesn't exist
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Ngram_Range", "Accuracy", "Precision", "Recall", "F1", "Training_Time"])
    
    # Append results to CSV
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        row = [
            model_name, 
            f"{ngram_range[0]}-{ngram_range[1]}",
            f"{metrics[0]:.5f}", 
            f"{metrics[1]:.5f}", 
            f"{metrics[2]:.5f}", 
            f"{metrics[3]:.5f}"
        ]
        # Add training time if provided
        row.append(f"{training_time:.2f}" if training_time is not None else "")
        writer.writerow(row)

def save_predictions(model_name, ngram_range, y_true, y_pred, predictions_dir="data/predictions"):
    """
    Save true and predicted labels for confusion matrix visualization.
    
    Saves predictions as numpy arrays for later analysis and confusion matrix generation.
    Creates two files: {model}_{ngram}_true.npy and {model}_{ngram}_pred.npy
    
    Args:
        model_name: Name of the model (spaces replaced with underscores in filename)
        ngram_range: Tuple of n-gram range, e.g., (1, 2)
        y_true: True labels (ground truth) from test set
        y_pred: Predicted labels from model
        predictions_dir: Directory to save prediction files (default: 'data/predictions')
    
    Example:
        >>> save_predictions('SVM', (1,2), y_test, y_pred)
        # Creates: data/predictions/SVM_1-2_true.npy and SVM_1-2_pred.npy
    """
    # Ensure predictions directory exists
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Create standardized filename (replace spaces with underscores)
    base_name = f"{model_name.replace(' ', '_')}_{ngram_range[0]}-{ngram_range[1]}"
    
    # Save true and predicted labels as numpy arrays
    np.save(f"{predictions_dir}/{base_name}_true.npy", y_true)
    np.save(f"{predictions_dir}/{base_name}_pred.npy", y_pred)
