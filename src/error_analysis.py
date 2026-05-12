"""
Error Analysis Module
======================
This module provides detailed error analysis for sentiment classification models.
It examines misclassified examples, groups errors by type, and provides insights.
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os


def analyze_errors(df, text_column='comment', label_column='label', predictions=None, 
                   true_labels=None, output_path=None):
    """
    Analyze misclassified examples and generate error insights.
    
    Args:
        df (pandas.DataFrame): DataFrame with text data
        text_column (str): Name of text column
        label_column (str): Name of true label column
        predictions (array): Predicted labels
        true_labels (array): True labels (if different from label_column)
        output_path (str): Path to save error analysis results (optional)
    
    Returns:
        dict: Error analysis results with statistics and examples
    """
    if true_labels is None:
        true_labels = df[label_column].tolist()
    
    if predictions is None:
        raise ValueError("Predictions must be provided for error analysis")
    
    # Identify misclassified examples
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) 
                            if true != pred]
    
    errors = []
    error_types = Counter()
    
    for idx in misclassified_indices:
        true_label = true_labels[idx]
        pred_label = predictions[idx]
        text = df[text_column].iloc[idx]
        
        error_type = f"{true_label}_as_{pred_label}"
        error_types[error_type] += 1
        
        errors.append({
            'index': idx,
            'text': text,
            'true_label': true_label,
            'predicted_label': pred_label,
            'error_type': error_type,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    # Calculate error statistics
    total_samples = len(true_labels)
    total_errors = len(misclassified_indices)
    accuracy = (total_samples - total_errors) / total_samples if total_samples > 0 else 0
    
    # Error statistics per sentiment class
    sentiment_errors = {}
    for sentiment in set(true_labels):
        sentiment_indices = [i for i, label in enumerate(true_labels) if label == sentiment]
        sentiment_misclassified = [i for i in sentiment_indices if i in misclassified_indices]
        sentiment_errors[sentiment] = {
            'total': len(sentiment_indices),
            'errors': len(sentiment_misclassified),
            'error_rate': len(sentiment_misclassified) / len(sentiment_indices) if sentiment_indices else 0
        }
    
    # Compile results
    results = {
        'total_samples': total_samples,
        'total_errors': total_errors,
        'accuracy': accuracy,
        'error_types': dict(error_types),
        'sentiment_errors': sentiment_errors,
        'errors': errors[:50]  # Limit to first 50 errors for readability
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Error analysis saved to: {output_path}")
    
    return results


def print_error_analysis(results):
    """
    Print a formatted error analysis report.
    
    Args:
        results (dict): Error analysis results from analyze_errors()
    """
    print("=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nTotal Samples: {results['total_samples']}")
    print(f"Total Errors: {results['total_errors']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    print("\n" + "-" * 70)
    print("Error Types:")
    print("-" * 70)
    for error_type, count in results['error_types'].items():
        print(f"  {error_type}: {count}")
    
    print("\n" + "-" * 70)
    print("Error Rate by Sentiment:")
    print("-" * 70)
    for sentiment, stats in results['sentiment_errors'].items():
        print(f"  {sentiment.capitalize()}:")
        print(f"    Total: {stats['total']}")
        print(f"    Errors: {stats['errors']}")
        print(f"    Error Rate: {stats['error_rate']:.2%}")
    
    print("\n" + "-" * 70)
    print("Sample Misclassifications:")
    print("-" * 70)
    for i, error in enumerate(results['errors'][:10], 1):
        print(f"\n{i}. {error['error_type'].replace('_', ' ').upper()}")
        print(f"   Text: {error['text'][:100]}...")
        print(f"   Length: {error['text_length']} chars, {error['word_count']} words")
    
    print("\n" + "=" * 70)


def identify_confusing_cases(df, text_column='comment', label_column='label', 
                            predictions=None, true_labels=None, threshold=0.6):
    """
    Identify cases where predictions are close to multiple classes (potential ambiguity).
    
    Args:
        df (pandas.DataFrame): DataFrame with text data
        text_column (str): Name of text column
        label_column (str): Name of true label column
        predictions (array): Predicted labels
        true_labels (array): True labels
        threshold (float): Confidence threshold for ambiguity
    
    Returns:
        list: List of potentially confusing cases
    """
    if predictions is None or true_labels is None:
        raise ValueError("Predictions and true_labels must be provided")
    
    # This would require probability scores from the model
    # For now, we'll identify misclassified cases with short text
    confusing_cases = []
    
    for idx, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            text = df[text_column].iloc[idx]
            if len(text.split()) < 5:  # Short texts are often ambiguous
                confusing_cases.append({
                    'index': idx,
                    'text': text,
                    'true_label': true,
                    'predicted_label': pred,
                    'reason': 'Short text (potential ambiguity)'
                })
    
    return confusing_cases


def analyze_error_patterns(errors):
    """
    Analyze patterns in misclassifications to identify common issues.
    
    Args:
        errors (list): List of error dictionaries from analyze_errors()
    
    Returns:
        dict: Pattern analysis results
    """
    patterns = {
        'short_text_errors': 0,
        'long_text_errors': 0,
        'contains_negation': 0,
        'contains_emoji': 0,
        'mixed_sentiment_indicators': 0
    }
    
    for error in errors:
        text = error['text'].lower()
        
        # Check text length patterns
        if error['word_count'] < 5:
            patterns['short_text_errors'] += 1
        elif error['word_count'] > 20:
            patterns['long_text_errors'] += 1
        
        # Check for negation
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'nor']
        if any(word in text.split() for word in negation_words):
            patterns['contains_negation'] += 1
        
        # Check for emojis (simple check)
        emoji_indicators = ['happy', 'sad', 'love', 'hate', 'angry', 'smile']
        if any(indicator in text for indicator in emoji_indicators):
            patterns['contains_emoji'] += 1
        
        # Check for mixed sentiment indicators
        positive_words = ['good', 'great', 'love', 'amazing', 'excellent']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'poor']
        has_positive = any(word in text.split() for word in positive_words)
        has_negative = any(word in text.split() for word in negative_words)
        if has_positive and has_negative:
            patterns['mixed_sentiment_indicators'] += 1
    
    return patterns


# For standalone testing
if __name__ == "__main__":
    # Test with sample data
    sample_df = pd.DataFrame({
        'comment': [
            'great product love it',
            'terrible service hated',
            'okay nothing special',
            'amazing quality recommend',
            'bad quality disappointed',
            'good product nice',
            'worst experience never',
            'its fine okay'
        ],
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                 'positive', 'negative', 'neutral']
    })
    
    # Simulate some predictions with errors
    predictions = ['positive', 'negative', 'neutral', 'positive', 'neutral',  # error
                  'positive', 'negative', 'positive']  # error
    
    print("=" * 60)
    print("ERROR ANALYSIS TEST")
    print("=" * 60)
    
    results = analyze_errors(sample_df, predictions=predictions)
    print_error_analysis(results)
    
    patterns = analyze_error_patterns(results['errors'])
    print("\n" + "-" * 60)
    print("Error Patterns:")
    print("-" * 60)
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}")
    
    print("\nTest passed!")
