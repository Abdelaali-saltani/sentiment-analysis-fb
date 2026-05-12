"""
Text Preprocessing Module
===========================
This module handles all text preprocessing steps for the sentiment analysis project.
It includes functions for cleaning and normalizing text data with advanced NLP techniques.
"""

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (only needed once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def remove_urls(text):
    """
    Remove URLs from text.
    
    Args:
        text (str): Input text containing potential URLs
        
    Returns:
        str: Text with URLs removed
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_punctuation(text):
    """
    Remove punctuation marks from text.
    
    Args:
        text (str): Input text with punctuation
        
    Returns:
        str: Text without punctuation
    """
    # Create translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    """
    Remove common English stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without stopwords
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text):
    """
    Convert words to their base form using lemmatization.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with lemmatized words
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def stem_text(text):
    """
    Convert words to their root form using stemming.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stemmed words
    """
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def handle_emojis(text):
    """
    Convert emojis to their text descriptions.
    
    Args:
        text (str): Input text potentially containing emojis
        
    Returns:
        str: Text with emojis converted to descriptions
    """
    # Common emoji mappings
    emoji_map = {
        '😀': 'happy_face',
        '😂': 'laughing_face',
        '😍': 'love_face',
        '🥰': 'love_face',
        '😊': 'smile_face',
        '😎': 'cool_face',
        '🤔': 'thinking_face',
        '😢': 'sad_face',
        '😭': 'crying_face',
        '😡': 'angry_face',
        '👍': 'thumbs_up',
        '👎': 'thumbs_down',
        '❤️': 'heart',
        '💔': 'broken_heart',
        '🔥': 'fire',
        '✨': 'sparkles',
        '🎉': 'party',
        '💯': 'hundred_points',
        '👏': 'clapping',
        '🙏': 'pray',
        '💪': 'muscle',
        '🤝': 'handshake',
    }
    
    for emoji, description in emoji_map.items():
        text = text.replace(emoji, f' {description} ')
    
    return text


def normalize_slang(text):
    """
    Normalize common slang and abbreviations to formal English.
    
    Args:
        text (str): Input text with potential slang
        
    Returns:
        str: Text with normalized slang
    """
    slang_map = {
        'u': 'you',
        'ur': 'your',
        'r': 'are',
        'n': 'and',
        'b4': 'before',
        'gr8': 'great',
        'lol': 'laugh_out_loud',
        'omg': 'oh_my_god',
        'idk': 'i_dont_know',
        'tbh': 'to_be_honest',
        'imo': 'in_my_opinion',
        'imho': 'in_my_humble_opinion',
        'btw': 'by_the_way',
        'fyi': 'for_your_information',
        'asap': 'as_soon_as_possible',
        'thx': 'thanks',
        'thnx': 'thanks',
        'tnx': 'thanks',
        'plz': 'please',
        'pls': 'please',
        'w/': 'with',
        'w/o': 'without',
        'b/c': 'because',
        'bc': 'because',
        'cuz': 'because',
        'cos': 'because',
        'gonna': 'going_to',
        'wanna': 'want_to',
        'gotta': 'got_to',
        'kinda': 'kind_of',
        'sorta': 'sort_of',
        'outta': 'out_of',
        'coulda': 'could_have',
        'shoulda': 'should_have',
        'woulda': 'would_have',
        'musta': 'must_have',
        'mighta': 'might_have',
    }
    
    words = text.split()
    normalized_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in slang_map:
            normalized_words.append(slang_map[word_lower])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def handle_negation(text):
    """
    Mark negated words to preserve negation context after stopword removal.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with negated words marked
    """
    negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing', 'nobody', 'nowhere'}
    words = word_tokenize(text)
    result = []
    negation = False
    
    for word in words:
        if word.lower() in negation_words:
            negation = True
            result.append(word)
        elif negation:
            # Mark the word as negated
            result.append(f'NOT_{word}')
            negation = False
        else:
            result.append(word)
    
    return ' '.join(result)


def preprocess_text(text, config=None):
    """
    Complete preprocessing pipeline for a single text with configurable options.
    
    Steps (configurable):
        1. Convert to lowercase
        2. Handle emojis
        3. Normalize slang
        4. Handle negation
        5. Remove URLs
        6. Remove punctuation
        7. Lemmatize or stem
        8. Remove stopwords
    
    Args:
        text (str): Raw input text
        config (dict): Configuration dictionary with boolean flags for each step
                       Default: all advanced features enabled
    
    Returns:
        str: Cleaned and preprocessed text
    """
    if config is None:
        config = {
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
    
    # Step 1: Convert to lowercase
    if config.get('lowercase', True):
        text = text.lower()
    
    # Step 2: Handle emojis
    if config.get('handle_emojis', True):
        text = handle_emojis(text)
    
    # Step 3: Normalize slang
    if config.get('normalize_slang', True):
        text = normalize_slang(text)
    
    # Step 4: Handle negation
    if config.get('handle_negation', True):
        text = handle_negation(text)
    
    # Step 5: Remove URLs
    if config.get('remove_urls', True):
        text = remove_urls(text)
    
    # Step 6: Remove punctuation
    if config.get('remove_punctuation', True):
        text = remove_punctuation(text)
    
    # Step 7: Lemmatize or stem (mutually exclusive)
    if config.get('lemmatize', True) and not config.get('stem', False):
        text = lemmatize_text(text)
    elif config.get('stem', False):
        text = stem_text(text)
    
    # Step 8: Remove stopwords
    if config.get('remove_stopwords', True):
        text = remove_stopwords(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_dataframe(df, text_column='comment', config=None):
    """
    Apply preprocessing to all text in a DataFrame with configurable options.
    
    Args:
        df (pandas.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to preprocess
        config (dict): Configuration dictionary for preprocessing steps
        
    Returns:
        pandas.DataFrame: DataFrame with preprocessed text
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Apply preprocessing to each comment
    df_processed[text_column] = df_processed[text_column].apply(
        lambda x: preprocess_text(str(x), config) if pd.notna(x) else ''
    )
    
    return df_processed


# For standalone testing
if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "This is a GREAT product! Check it out at https://example.com",
        "I hate this... it's so bad!!!",
        "It's okay, nothing special.",
        "Visit www.facebook.com for more info!",
    ]
    
    print("=" * 60)
    print("TEXT PREPROCESSING TEST")
    print("=" * 60)
    
    for text in sample_texts:
        print(f"\nOriginal: {text}")
        print(f"Processed: {preprocess_text(text)}")
        print("-" * 60)
