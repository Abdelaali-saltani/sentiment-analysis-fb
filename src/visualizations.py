"""
NLP Visualizations Module
==========================
This module creates various visualizations for NLP analysis including word clouds,
confusion matrices, feature importance plots, and sentiment analysis charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os


def create_word_cloud(text, title="Word Cloud", output_path=None):
    """
    Create and display a word cloud from text.
    
    Args:
        text (str): Text to generate word cloud from
        title (str): Title for the word cloud
        output_path (str): Path to save the word cloud image (optional)
    
    Returns:
        WordCloud: The generated word cloud object
    """
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout(pad=0)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to: {output_path}")
    
    plt.close()
    return wordcloud


def create_sentiment_word_clouds(df, text_column='comment', label_column='label', output_dir='models/visualizations'):
    """
    Create word clouds for each sentiment class.
    
    Args:
        df (pandas.DataFrame): DataFrame with text and labels
        text_column (str): Name of text column
        label_column (str): Name of label column
        output_dir (str): Directory to save visualizations
    """
    sentiments = df[label_column].unique()
    
    for sentiment in sentiments:
        sentiment_text = ' '.join(df[df[label_column] == sentiment][text_column].tolist())
        
        if len(sentiment_text) > 0:
            output_path = os.path.join(output_dir, f'wordcloud_{sentiment}.png')
            create_word_cloud(
                sentiment_text,
                title=f'{sentiment.capitalize()} Sentiment Word Cloud',
                output_path=output_path
            )


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', output_path=None):
    """
    Plot a confusion matrix with annotations.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        labels (list): List of label names
        title (str): Title for the plot
        output_path (str): Path to save the plot (optional)
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    
    plt.close()


def plot_feature_importance(feature_names, importance_scores, top_n=20, title='Feature Importance', output_path=None):
    """
    Plot feature importance scores.
    
    Args:
        feature_names (list): List of feature names
        importance_scores (array): Importance scores for each feature
        top_n (int): Number of top features to display
        title (str): Title for the plot
        output_path (str): Path to save the plot (optional)
    """
    # Get top features
    indices = np.argsort(importance_scores)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_scores = [importance_scores[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_scores, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {output_path}")
    
    plt.close()


def plot_sentiment_distribution(df, label_column='label', title='Sentiment Distribution', output_path=None):
    """
    Plot the distribution of sentiment classes.
    
    Args:
        df (pandas.DataFrame): DataFrame with labels
        label_column (str): Name of label column
        title (str): Title for the plot
        output_path (str): Path to save the plot (optional)
    """
    sentiment_counts = df[label_column].value_counts()
    
    plt.figure(figsize=(8, 6))
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    bar_colors = [colors.get(label, '#3498db') for label in sentiment_counts.index]
    
    sentiment_counts.plot(kind='bar', color=bar_colors)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Sentiment distribution plot saved to: {output_path}")
    
    plt.close()


def plot_text_length_vs_sentiment(df, text_column='comment', label_column='label', title='Text Length vs Sentiment', output_path=None):
    """
    Plot text length distribution by sentiment class.
    
    Args:
        df (pandas.DataFrame): DataFrame with text and labels
        text_column (str): Name of text column
        label_column (str): Name of label column
        title (str): Title for the plot
        output_path (str): Path to save the plot (optional)
    """
    df['text_length'] = df[text_column].str.len()
    
    plt.figure(figsize=(10, 6))
    sentiments = df[label_column].unique()
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    
    for sentiment in sentiments:
        sentiment_data = df[df[label_column] == sentiment]['text_length']
        plt.hist(sentiment_data, alpha=0.5, label=sentiment.capitalize(), 
                 color=colors.get(sentiment, '#3498db'), bins=30)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Text length plot saved to: {output_path}")
    
    plt.close()


def generate_all_visualizations(df, text_column='comment', label_column='label', y_true=None, y_pred=None, 
                                 feature_names=None, importance_scores=None, output_dir='models/visualizations'):
    """
    Generate all NLP visualizations at once.
    
    Args:
        df (pandas.DataFrame): DataFrame with text and labels
        text_column (str): Name of text column
        label_column (str): Name of label column
        y_true (array): True labels for confusion matrix (optional)
        y_pred (array): Predicted labels for confusion matrix (optional)
        feature_names (list): Feature names for importance plot (optional)
        importance_scores (array): Importance scores (optional)
        output_dir (str): Directory to save visualizations
    """
    print("Generating NLP visualizations...")
    print("-" * 60)
    
    # Word clouds for each sentiment
    create_sentiment_word_clouds(df, text_column, label_column, output_dir)
    
    # Sentiment distribution
    plot_sentiment_distribution(df, label_column, 
                                 output_path=os.path.join(output_dir, 'sentiment_distribution.png'))
    
    # Text length vs sentiment
    plot_text_length_vs_sentiment(df, text_column, label_column,
                                  output_path=os.path.join(output_dir, 'text_length_vs_sentiment.png'))
    
    # Confusion matrix (if predictions provided)
    if y_true is not None and y_pred is not None:
        labels = sorted(set(y_true) | set(y_pred))
        plot_confusion_matrix(y_true, y_pred, labels,
                             output_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Feature importance (if provided)
    if feature_names is not None and importance_scores is not None:
        plot_feature_importance(feature_names, importance_scores,
                               output_path=os.path.join(output_dir, 'feature_importance.png'))
    
    print("-" * 60)
    print("All visualizations generated successfully!")


# For standalone testing
if __name__ == "__main__":
    # Test with sample data
    sample_df = pd.DataFrame({
        'comment': [
            'great product love it amazing quality',
            'terrible service hated experience bad',
            'okay nothing special average',
            'amazing quality highly recommend love',
            'bad quality disappointed terrible',
            'good product nice quality happy',
            'worst experience never again hate',
            'its fine nothing special okay'
        ] * 10,
        'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                 'positive', 'negative', 'neutral'] * 10
    })
    
    print("=" * 60)
    print("NLP VISUALIZATIONS TEST")
    print("=" * 60)
    
    generate_all_visualizations(sample_df, output_dir='models/visualizations')
    
    print("Test passed!")
