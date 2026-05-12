"""
Text Analysis Module
=====================
This module provides text analysis features including keyword extraction,
sentiment word highlighting, and text complexity metrics.
"""

import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Sentiment lexicons
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
    'love', 'like', 'happy', 'pleased', 'satisfied', 'delighted', 'thrilled',
    'best', 'better', 'nice', 'beautiful', 'perfect', 'brilliant', 'outstanding',
    'superb', 'magnificent', 'terrific', 'fabulous', 'recommend', 'recommended',
    'impressive', 'enjoy', 'enjoyed', 'pleasure', 'glad', 'excited', 'positive',
    'success', 'successful', 'win', 'winner', 'benefit', 'beneficial', 'helpful',
    'useful', 'effective', 'efficient', 'quality', 'high quality', 'top quality'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'worse', 'hate',
    'dislike', 'sad', 'disappointed', 'unhappy', 'frustrated', 'angry', 'annoyed',
    'upset', 'boring', 'dull', 'boring', 'useless', 'waste', 'wasted', 'failed',
    'failure', 'problem', 'problematic', 'issue', 'trouble', 'difficult', 'hard',
    'negative', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 'none',
    'complain', 'complaint', 'complaining', 'regret', 'sorry', 'apologize', 'mistake',
    'error', 'wrong', 'incorrect', 'broken', 'damage', 'damaged', 'slow', 'slowly'
}


def extract_keywords(texts, top_n=10):
    """
    Extract top keywords from texts using TF-IDF.
    
    Args:
        texts (list): List of text strings
        top_n (int): Number of top keywords to extract
    
    Returns:
        list: List of (keyword, score) tuples
    """
    vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Get feature names and sum TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Create and sort keyword scores
    keyword_scores = list(zip(feature_names, tfidf_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]


def extract_keywords_by_sentiment(df, text_column='comment', label_column='label', top_n=10):
    """
    Extract top keywords for each sentiment class.
    
    Args:
        df (pandas.DataFrame): DataFrame with text and labels
        text_column (str): Name of text column
        label_column (str): Name of label column
        top_n (int): Number of top keywords per sentiment
    
    Returns:
        dict: Dictionary mapping sentiment to list of (keyword, score) tuples
    """
    keywords_by_sentiment = {}
    
    for sentiment in df[label_column].unique():
        sentiment_texts = df[df[label_column] == sentiment][text_column].tolist()
        keywords = extract_keywords(sentiment_texts, top_n)
        keywords_by_sentiment[sentiment] = keywords
    
    return keywords_by_sentiment


def highlight_sentiment_words(text, positive_color='#2ecc71', negative_color='#e74c3c'):
    """
    Highlight positive and negative words in text with HTML spans.
    
    Args:
        text (str): Input text
        positive_color (str): Color for positive word highlighting
        negative_color (str): Color for negative word highlighting
    
    Returns:
        str: Text with highlighted sentiment words
    """
    words = text.split()
    highlighted_words = []
    
    for word in words:
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in POSITIVE_WORDS:
            highlighted_words.append(f'<span style="color: {positive_color}; font-weight: bold;">{word}</span>')
        elif clean_word in NEGATIVE_WORDS:
            highlighted_words.append(f'<span style="color: {negative_color}; font-weight: bold;">{word}</span>')
        else:
            highlighted_words.append(word)
    
    return ' '.join(highlighted_words)


def get_sentiment_word_counts(text):
    """
    Count positive and negative words in text.
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Dictionary with positive and negative word counts
    """
    words = text.lower().split()
    words = [re.sub(r'[^\w]', '', word) for word in words]
    
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    return {
        'positive_words': positive_count,
        'negative_words': negative_count,
        'total_words': len(words)
    }


def get_most_frequent_words(texts, n=20):
    """
    Get the most frequent words in a list of texts.
    
    Args:
        texts (list): List of text strings
        n (int): Number of most frequent words to return
    
    Returns:
        list: List of (word, count) tuples
    """
    all_words = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(n)


def calculate_text_complexity(text):
    """
    Calculate text complexity metrics.
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Dictionary with complexity metrics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Syllable count (approximate)
    syllable_count = sum(count_syllables(word) for word in words)
    
    # Flesch Reading Ease (approximate)
    if sentence_count > 0 and word_count > 0:
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / word_count))
    else:
        flesch_score = 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'syllable_count': syllable_count,
        'flesch_reading_ease': round(flesch_score, 2)
    }


def count_syllables(word):
    """
    Count syllables in a word (approximate).
    
    Args:
        word (str): Input word
    
    Returns:
        int: Approximate syllable count
    """
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    word = re.sub(r'(es|ed)$', '', word)
    vowels = r'[aeiouy]+'
    syllable_count = len(re.findall(vowels, word))
    
    if syllable_count == 0:
        syllable_count = 1
    
    return syllable_count


def analyze_text(text):
    """
    Perform comprehensive text analysis.
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Comprehensive text analysis results
    """
    sentiment_counts = get_sentiment_word_counts(text)
    complexity = calculate_text_complexity(text)
    highlighted = highlight_sentiment_words(text)
    
    return {
        'sentiment_word_counts': sentiment_counts,
        'complexity_metrics': complexity,
        'highlighted_text': highlighted,
        'sentiment_ratio': sentiment_counts['positive_words'] - sentiment_counts['negative_words']
    }


# For standalone testing
if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "This is a great product! I love it so much. Amazing quality!",
        "Terrible service. I hate this experience. Very disappointed.",
        "It's okay, nothing special really. Average quality."
    ]
    
    print("=" * 60)
    print("TEXT ANALYSIS TEST")
    print("=" * 60)
    
    # Test keyword extraction
    print("\nTop Keywords:")
    keywords = extract_keywords(sample_texts, top_n=5)
    for word, score in keywords:
        print(f"  {word}: {score:.4f}")
    
    # Test sentiment highlighting
    print("\nSentiment Word Highlighting:")
    for text in sample_texts:
        highlighted = highlight_sentiment_words(text)
        print(f"  Original: {text}")
        print(f"  Highlighted: {highlighted}")
        print()
    
    # Test sentiment word counts
    print("Sentiment Word Counts:")
    for text in sample_texts:
        counts = get_sentiment_word_counts(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Positive: {counts['positive_words']}, Negative: {counts['negative_words']}")
    
    # Test complexity metrics
    print("\nText Complexity:")
    for text in sample_texts:
        complexity = calculate_text_complexity(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Words: {complexity['word_count']}, Sentences: {complexity['sentence_count']}")
        print(f"  Flesch Score: {complexity['flesch_reading_ease']}")
    
    print("\nTest passed!")
