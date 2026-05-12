"""
Model Evaluation Module
=======================
This module handles evaluation of the trained sentiment classification model.
It provides various metrics and visualization capabilities.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true, y_pred, average='weighted'):
    """
    Calculate precision score.
    
    Precision = True Positives / (True Positives + False Positives)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Method to calculate average ('weighted', 'macro', 'micro')
        
    Returns:
        float: Precision score
    """
    return precision_score(y_true, y_pred, average=average)


def calculate_recall(y_true, y_pred, average='weighted'):
    """
    Calculate recall score.
    
    Recall = True Positives / (True Positives + False Negatives)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Method to calculate average
        
    Returns:
        float: Recall score
    """
    return recall_score(y_true, y_pred, average=average)


def calculate_f1(y_true, y_pred, average='weighted'):
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Method to calculate average
        
    Returns:
        float: F1 score
    """
    return f1_score(y_true, y_pred, average=average)


def calculate_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix
        
    Returns:
        ndarray: Confusion matrix
    """
    if labels is None:
        labels = ['negative', 'neutral', 'positive']
    
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_confusion_matrix(cm, labels=None):
    """
    Pretty print the confusion matrix.
    
    Args:
        cm: Confusion matrix
        labels: List of class labels
    """
    if labels is None:
        labels = ['negative', 'neutral', 'positive']
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    
    # Print header
    header = "Actual \\ Predicted |"
    for label in labels:
        header += f" {label:>10} |"
    print(header)
    print("-" * 40)
    
    # Print rows
    for i, label in enumerate(labels):
        row = f"{label:>15} |"
        for j in range(len(labels)):
            row += f" {cm[i][j]:>10} |"
        print(row)
    print("-" * 40)


def evaluate_model(y_true, y_pred, labels=None):
    """
    Complete model evaluation with all metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    if labels is None:
        labels = ['negative', 'neutral', 'positive']
    
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Calculate metrics
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, average='weighted')
    recall = calculate_recall(y_true, y_pred, average='weighted')
    f1 = calculate_f1(y_true, y_pred, average='weighted')
    
    # Print results
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = calculate_confusion_matrix(y_true, y_pred, labels)
    print_confusion_matrix(cm, labels)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    }
    
    return metrics


def get_sentiment_statistics(predictions):
    """
    Calculate sentiment distribution statistics.
    
    Args:
        predictions (list): List of predicted sentiments
        
    Returns:
        dict: Dictionary with sentiment counts and percentages
    """
    total = len(predictions)
    
    counts = {
        'positive': predictions.count('positive'),
        'negative': predictions.count('negative'),
        'neutral': predictions.count('neutral')
    }
    
    percentages = {
        'positive': (counts['positive'] / total * 100) if total > 0 else 0,
        'negative': (counts['negative'] / total * 100) if total > 0 else 0,
        'neutral': (counts['neutral'] / total * 100) if total > 0 else 0
    }
    
    return {
        'counts': counts,
        'percentages': percentages,
        'total': total
    }


# For standalone testing
if __name__ == "__main__":
    # Test with dummy predictions
    y_true = ['positive', 'positive', 'negative', 'neutral', 
              'negative', 'positive', 'neutral', 'negative', 'positive', 'neutral']
    y_pred = ['positive', 'positive', 'negative', 'negative', 
              'negative', 'positive', 'neutral', 'positive', 'positive', 'neutral']
    
    print("=" * 60)
    print("EVALUATION MODULE TEST")
    print("=" * 60)
    
    # Run evaluation
    metrics = evaluate_model(y_true, y_pred)
    
    # Test statistics
    print("\nSentiment Statistics:")
    stats = get_sentiment_statistics(y_pred)
    for sentiment, count in stats['counts'].items():
        print(f"  {sentiment}: {count} ({stats['percentages'][sentiment]:.1f}%)")
    
    print("\nTest passed!")
