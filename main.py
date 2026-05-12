"""
Sentiment Analysis Training Pipeline
====================================
This is the main training script for the Facebook Comments Sentiment Analysis project.
It orchestrates the entire training process from data loading to model saving.

Usage:
    python main.py

Steps:
    1. Load dataset
    2. Preprocess text
    3. Extract TF-IDF features
    4. Train Logistic Regression model
    5. Evaluate model performance
    6. Save model and vectorizer
"""

import pandas as pd
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_dataframe
from features import create_vectorizer, fit_transform_text, save_vectorizer
from train import split_data, create_model, train_model, save_model, predict
from evaluate import evaluate_model
from visualizations import generate_all_visualizations
from error_analysis import analyze_errors, print_error_analysis


def load_dataset(filepath):
    """
    Load the dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found at {filepath}")
        print("\nPlease ensure your dataset file exists with columns: comment, label")
        print("\nExpected format:")
        print("comment,label")
        print("I love this product!,positive")
        print("Terrible experience,negative")
        print("It's okay,neutral")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    if 'comment' not in df.columns or 'label' not in df.columns:
        print("Error: Dataset must contain 'comment' and 'label' columns")
        sys.exit(1)
    
    # Check label values
    valid_labels = {'positive', 'negative', 'neutral'}
    unique_labels = set(df['label'].unique())
    invalid_labels = unique_labels - valid_labels
    
    if invalid_labels:
        print(f"Warning: Found invalid labels: {invalid_labels}")
        print(f"Valid labels are: {valid_labels}")
    
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print()
    
    return df


def main():
    """
    Main training pipeline.
    """
    print("=" * 70)
    print("FACEBOOK COMMENTS SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Define file paths
    data_path = os.path.join('data', 'raw', 'comments.csv')
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    # Step 1: Load Dataset
    df = load_dataset(data_path)
    
    # Step 2: Preprocess Text
    print("Step 1: Preprocessing text data...")
    print("-" * 70)
    
    # Use advanced NLP preprocessing configuration
    preprocess_config = {
        'lowercase': True,
        'handle_emojis': True,
        'normalize_slang': True,
        'handle_negation': True,
        'remove_urls': True,
        'remove_punctuation': True,
        'lemmatize': True,
        'stem': False,
        'remove_stopwords': True,
    }
    
    df_processed = preprocess_dataframe(df, text_column='comment', config=preprocess_config)
    print("Preprocessing completed!")
    print(f"Preprocessing config: {preprocess_config}\n")
    
    # Show sample of preprocessed data
    print("Sample preprocessed data:")
    for i in range(min(3, len(df_processed))):
        print(f"  Original: {df['comment'].iloc[i][:80]}...")
        print(f"  Processed: {df_processed['comment'].iloc[i][:80]}...")
        print(f"  Label: {df_processed['label'].iloc[i]}")
        print()
    
    # Step 3: Feature Extraction (TF-IDF)
    print("Step 2: Extracting TF-IDF features...")
    print("-" * 70)
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(max_features=5000)
    X = fit_transform_text(vectorizer, df_processed['comment'].tolist())
    y = df_processed['label'].tolist()
    print("Feature extraction completed!\n")
    
    # Step 4: Split Data
    print("Step 3: Splitting data (80% train, 20% test)...")
    print("-" * 70)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print()
    
    # Step 5: Train Model
    print("Step 4: Training Logistic Regression model...")
    print("-" * 70)
    model = create_model(max_iter=1000, C=1.0, random_state=42)
    model = train_model(model, X_train, y_train)
    print()
    
    # Step 6: Evaluate Model
    print("Step 5: Evaluating model performance...")
    print("-" * 70)
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    print()
    
    # Step 7: Generate NLP Visualizations
    print("Step 6: Generating NLP visualizations...")
    print("-" * 70)
    try:
        # Get feature names and importance scores for Logistic Regression
        feature_names = vectorizer.get_feature_names_out()
        importance_scores = model.coef_[0]  # For Logistic Regression
        importance_scores = np.abs(importance_scores)  # Use absolute values
        
        generate_all_visualizations(
            df_processed,
            text_column='comment',
            label_column='label',
            y_true=y_test,
            y_pred=y_pred,
            feature_names=feature_names,
            importance_scores=importance_scores,
            output_dir='models/visualizations'
        )
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")
    print()
    
    # Step 8: Error Analysis
    print("Step 7: Performing error analysis...")
    print("-" * 70)
    try:
        error_results = analyze_errors(
            df_processed,
            text_column='comment',
            label_column='label',
            predictions=y_pred,
            true_labels=y_test,
            output_path='models/error_analysis.json'
        )
        print_error_analysis(error_results)
    except Exception as e:
        print(f"Warning: Error analysis failed: {e}")
    print()
    
    # Step 9: Save Model and Vectorizer
    print("Step 8: Saving model and vectorizer...")
    print("-" * 70)
    save_model(model, model_path)
    save_vectorizer(vectorizer, vectorizer_path)
    print()
    
    # Summary
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"\nFinal Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Final F1-Score: {metrics['f1_score']:.4f}")
    print("\nNext step: Run 'python app.py' to start the web interface")
    print("=" * 70)


if __name__ == "__main__":
    main()
