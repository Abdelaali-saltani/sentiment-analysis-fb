"""
Model Training Module
=====================
This module handles training of the sentiment classification model.
Supports multiple NLP-specific classifiers including Logistic Regression,
Naive Bayes, and SVM.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib
import os


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix (TF-IDF vectors)
        y: Target labels
        test_size (float): Proportion of data to use for testing (default: 0.2 = 20%)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def create_model(max_iter=1000, C=1.0, random_state=42, model_type='logistic'):
    """
    Create a sentiment classification model.
    
    Parameters:
        max_iter (int): Maximum number of iterations for convergence
        C (float): Inverse of regularization strength (smaller = stronger regularization)
        random_state (int): Random seed for reproducibility
        model_type (str): Type of model ('logistic', 'naive_bayes', 'svm')
        
    Returns:
        Configured model
    """
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
    elif model_type == 'naive_bayes':
        model = MultinomialNB(alpha=1.0)
    elif model_type == 'svm':
        model = LinearSVC(
            C=C,
            random_state=random_state,
            class_weight='balanced',
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model(model, X_train, y_train):
    """
    Train the model on the training data.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        
    Returns:
        trained_model: The trained model
    """
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model


def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        filepath (str): Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save using joblib
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """
    Load a saved model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        array: Predicted labels
    """
    return model.predict(X)


def predict_proba(model, X):
    """
    Get prediction probabilities for each class.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        array: Prediction probabilities
    """
    return model.predict_proba(X)


# For standalone testing
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    
    print("=" * 60)
    print("MODEL TRAINING TEST")
    print("=" * 60)
    
    # Create dummy data for testing
    X, y = make_classification(
        n_samples=1000, 
        n_features=100, 
        n_classes=3, 
        n_informative=50,
        random_state=42
    )
    
    # Convert to string labels for sentiment
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    y = [label_map[label] for label in y]
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create and train model
    model = create_model()
    model = train_model(model, X_train, y_train)
    
    # Make predictions
    predictions = predict(model, X_test[:5])
    print(f"\nSample predictions: {predictions}")
    
    print("Test passed!")
