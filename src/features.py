"""
Feature Extraction Module
==========================
This module handles TF-IDF feature extraction for text data with multiple methods.
TF-IDF (Term Frequency - Inverse Document Frequency) converts text into numerical vectors.
Supports n-grams, character n-grams, and additional text features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import numpy as np


def create_vectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2), analyzer='word'):
    """
    Create a TF-IDF vectorizer with specified parameters.
    
    Parameters:
        max_features (int): Maximum number of features (vocabulary size)
        min_df (int): Minimum document frequency (ignore terms appearing in fewer documents)
        max_df (float): Maximum document frequency (ignore terms appearing in more than X% of documents)
        ngram_range (tuple): Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
        analyzer (str): 'word' for word n-grams, 'char' for character n-grams
    
    Returns:
        TfidfVectorizer: Configured TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,    # Limit vocabulary size
        min_df=min_df,                # Ignore rare terms
        max_df=max_df,                # Ignore overly common terms
        ngram_range=ngram_range,      # N-gram range
        analyzer=analyzer,            # Word or character level
        sublinear_tf=True             # Apply sublinear tf scaling
    )
    
    return vectorizer


def create_word_ngram_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """
    Create a TF-IDF vectorizer for word n-grams.
    
    Parameters:
        max_features (int): Maximum number of features
        ngram_range (tuple): N-gram range (e.g., (1, 3) for unigrams, bigrams, trigrams)
    
    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    return create_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer='word'
    )


def create_char_ngram_vectorizer(max_features=5000, ngram_range=(3, 5)):
    """
    Create a TF-IDF vectorizer for character n-grams.
    Character n-grams capture morphological patterns and are robust to misspellings.
    
    Parameters:
        max_features (int): Maximum number of features
        ngram_range (tuple): Character n-gram range (e.g., (3, 5) for 3-5 character sequences)
    
    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    return create_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer='char'
    )


def extract_text_features(texts):
    """
    Extract additional text features (word count, text length, etc.).
    
    Parameters:
        texts (list): List of text strings
    
    Returns:
        numpy.ndarray: Feature matrix with additional text features
    """
    features = []
    
    for text in texts:
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        features.append([word_count, char_count, avg_word_length])
    
    return np.array(features)


def fit_transform_text(vectorizer, texts):
    """
    Fit the vectorizer on texts and transform them into TF-IDF features.
    
    Args:
        vectorizer (TfidfVectorizer): The vectorizer to fit
        texts (list): List of preprocessed text strings
        
    Returns:
        scipy.sparse matrix: TF-IDF feature matrix
    """
    # Fit and transform the texts
    X = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X


def transform_text(vectorizer, texts):
    """
    Transform texts using a fitted vectorizer.
    Use this for new data (e.g., predictions).
    
    Args:
        vectorizer (TfidfVectorizer): Pre-fitted vectorizer
        texts (list): List of preprocessed text strings
        
    Returns:
        scipy.sparse matrix: TF-IDF feature matrix
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, filepath):
    """
    Save the fitted vectorizer to disk.
    
    Args:
        vectorizer (TfidfVectorizer): Fitted vectorizer to save
        filepath (str): Path where to save the vectorizer
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save using joblib
    joblib.dump(vectorizer, filepath)
    print(f"Vectorizer saved to: {filepath}")


def load_vectorizer(filepath):
    """
    Load a saved vectorizer from disk.
    
    Args:
        filepath (str): Path to the saved vectorizer
        
    Returns:
        TfidfVectorizer: Loaded vectorizer
    """
    vectorizer = joblib.load(filepath)
    print(f"Vectorizer loaded from: {filepath}")
    return vectorizer


# For standalone testing
if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "great product love it",
        "terrible service hated experience",
        "okay nothing special",
        "amazing quality highly recommend",
        "bad quality disappointed"
    ]
    
    print("=" * 60)
    print("TF-IDF FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(max_features=100)
    X = fit_transform_text(vectorizer, sample_texts)
    
    print("\nFeature names (top 10):")
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names[:10])
    
    print(f"\nFeature matrix shape: {X.shape}")
    print("Test passed!")
