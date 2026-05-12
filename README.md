# Sentiment Analysis of Facebook Comments

A complete machine learning project for classifying social media comments into **Positive**, **Negative**, and **Neutral** sentiments using Python and scikit-learn.

![Project Banner](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)

## Project Overview

This project demonstrates a complete NLP pipeline for sentiment analysis with advanced features:

- **Advanced Text Preprocessing**: Lowercasing, URL removal, punctuation removal, stopword removal, lemmatization, stemming, emoji handling, slang normalization, negation detection
- **Feature Extraction**: TF-IDF vectorization with n-grams (unigrams, bigrams, trigrams), character n-grams, and additional text features
- **Machine Learning**: Multiple NLP-specific classifiers (Logistic Regression, Naive Bayes, SVM)
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, Error Analysis
- **NLP Visualizations**: Word clouds, confusion matrices, feature importance plots, sentiment distribution charts
- **Text Analysis**: Keyword extraction, sentiment word highlighting, text complexity metrics
- **Web Interface**: Flask-based UI for real-time predictions with NLP analysis dashboard

## Folder Structure

```
sentiment-analysis-fb/
│
├── data/
│   └── raw/              # Raw CSV data
│       └── comments.csv  # Training dataset
│
├── models/               # Saved models and vectorizers
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── error_analysis.json
│   └── visualizations/   # Generated NLP visualizations
│       ├── wordcloud_positive.png
│       ├── wordcloud_negative.png
│       ├── wordcloud_neutral.png
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── sentiment_distribution.png
│       └── text_length_vs_sentiment.png
│
├── src/                  # Source code modules
│   ├── preprocess.py     # Advanced text preprocessing
│   ├── features.py       # TF-IDF feature extraction with n-grams
│   ├── train.py          # Model training (multiple classifiers)
│   ├── evaluate.py       # Model evaluation
│   ├── visualizations.py # NLP visualizations
│   ├── error_analysis.py # Error analysis module
│   └── text_analysis.py  # Text analysis features
│
├── templates/            # Flask HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Home page with analyzer
│   ├── analytics.html    # Analytics dashboard
│   ├── analysis.html     # NLP Analysis dashboard
│   └── about.html        # About page
│
├── static/               # CSS, JS, images
│   └── style.css         # Main stylesheet
│
├── main.py               # Training pipeline script
├── app.py                # Flask web application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Clone or Download the Project

```bash
cd sentiment-analysis-fb
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

The preprocessing script will automatically download required NLTK data on first run.

## Dataset Format

Create a CSV file at `data/raw/comments.csv` with the following format:

```csv
comment,label
I love this product! It's amazing!,positive
This is terrible and I hate it,negative
It's okay I guess nothing special,neutral
Best purchase ever highly recommend,positive
Worst experience ever never again,negative
```

**Requirements:**
- Two columns: `comment` and `label`
- Labels must be one of: `positive`, `negative`, `neutral`
- At least 20-30 samples for basic training (more is better)

## Usage

### Step 1: Train the Model

Run the training pipeline to preprocess data, extract features, train the model, and save it:

```bash
python main.py
```

This will:
1. Load and preprocess the dataset with advanced NLP preprocessing
2. Extract TF-IDF features with n-grams
3. Split data (80% train, 20% test)
4. Train Logistic Regression model
5. Evaluate and display metrics
6. Generate NLP visualizations (word clouds, confusion matrix, feature importance, etc.)
7. Perform error analysis and save results
8. Save model to `models/sentiment_model.pkl`
9. Save vectorizer to `models/tfidf_vectorizer.pkl`

**Expected Output:**
```
============================================================
FACEBOOK COMMENTS SENTIMENT ANALYSIS - TRAINING PIPELINE
============================================================

Step 1: Preprocessing text data...
------------------------------------------------------------
Preprocessing completed!

Step 2: Extracting TF-IDF features...
------------------------------------------------------------
TF-IDF matrix shape: (1000, 5000)
Feature extraction completed!

Step 3: Splitting data (80% train, 20% test)...
------------------------------------------------------------
Training set size: 800
Testing set size: 200

Step 4: Training Logistic Regression model...
------------------------------------------------------------
Training completed!

Step 5: Evaluating model performance...
------------------------------------------------------------
Accuracy:  0.8500 (85.00%)
Precision: 0.8480
Recall:    0.8500
F1-Score:  0.8485

Step 6: Saving model and vectorizer...
------------------------------------------------------------
Model saved to: models\sentiment_model.pkl
Vectorizer saved to: models\tfidf_vectorizer.pkl

============================================================
TRAINING COMPLETED SUCCESSFULLY!
============================================================
```

### Step 2: Run the Web Application

After training, start the Flask web server:

```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

### Web Interface Features

The web interface includes multiple pages and features:

#### Home Page (`/`)
- **Hero Section**: Project overview with key statistics (samples, accuracy, response time)
- **Analyzer Section**: Enter comments for sentiment analysis with real-time predictions
  - Single or batch comment analysis
  - Confidence scores for each sentiment class
  - Sentiment word highlighting (positive in green, negative in red)
  - Text complexity metrics
- **Results Section**: Display analysis results with sentiment distribution
- **Use Cases Section**: 6 animated use case cards showcasing real-world applications
  - Social Media Monitoring
  - Customer Feedback Analysis
  - Product Review Analysis
  - Market Research
  - Content Moderation
  - Survey Analysis
- **Tutorial Section**: Animated step-by-step guide on how to use the analyzer

#### Analytics Dashboard (`/analytics`)
- **Dataset Statistics**: Total samples, model accuracy, algorithm info
- **Class Distribution**: Visual breakdown of sentiment classes
- **Advanced Analytics**:
  - Sentiment Trends Over Time (with time range selector)
  - Confidence Distribution chart
  - Model Performance metrics (Precision, Recall, F1-score per class)
- **Word Frequency Analysis**: Word clouds for each sentiment class
- **Performance Metrics**: Processing speed, precision by class, model performance chart

#### NLP Analysis Dashboard (`/analysis`)
- **Sentiment Word Clouds**: Visual representation of most frequent words per sentiment
- **Confusion Matrix**: Visual representation of model predictions vs actual labels
- **Feature Importance**: Top features (words) for sentiment classification
- **Sentiment Distribution**: Class distribution in the training dataset
- **Text Length vs Sentiment**: Analysis of text length distribution across sentiment classes
- **Preprocessing Pipeline Info**: Details about the preprocessing configuration used

#### About Page (`/about`)
- Project information and technical details

## API Usage

You can also use the API programmatically:

```python
import requests

# Single comment
response = requests.post('http://localhost:5000/analyze', 
    json={'text': 'This product is amazing!'})
print(response.json())

# Multiple comments
text = '''This is great!
I hate this product.
It's okay I guess.'''

response = requests.post('http://localhost:5000/analyze', 
    json={'text': text})
print(response.json())
```

**Sample API Response:**
```json
{
    "original_text": "This product is amazing!",
    "processed_text": "product amazing",
    "sentiment": "positive",
    "confidence": 92.45,
    "all_confidences": {
        "negative": 3.21,
        "neutral": 4.34,
        "positive": 92.45
    },
    "highlighted_text": "This product is <span class='positive-word'>amazing</span>!",
    "text_analysis": {
        "word_count": 4,
        "char_count": 24,
        "avg_word_length": 6.0,
        "sentiment_words": {
            "positive": ["amazing"],
            "negative": []
        },
        "complexity": {
            "flesch_reading_ease": 82.3,
            "avg_sentence_length": 24.0
        }
    }
}
```

## Module Details

### src/preprocess.py
Advanced text preprocessing functions:
- `preprocess_text()` - Full preprocessing pipeline with configurable options
- `remove_urls()` - Remove URLs
- `remove_punctuation()` - Remove punctuation
- `remove_stopwords()` - Remove common stopwords
- `lemmatize_text()` - Convert words to base form using WordNetLemmatizer
- `stem_text()` - Convert words to root form using PorterStemmer
- `handle_emojis()` - Convert emojis to text descriptions
- `normalize_slang()` - Normalize common slang and abbreviations
- `handle_negation()` - Mark negated words to preserve context

### src/features.py
TF-IDF feature extraction with multiple methods:
- `create_vectorizer()` - Create TF-IDF vectorizer with n-grams
- `create_word_ngram_vectorizer()` - Create word n-gram vectorizer
- `create_char_ngram_vectorizer()` - Create character n-gram vectorizer
- `extract_text_features()` - Extract additional text features (word count, length)
- `fit_transform_text()` - Fit and transform training data
- `save_vectorizer()` / `load_vectorizer()` - Persist vectorizer

### src/train.py
Model training with multiple classifiers:
- `create_model()` - Create model (supports 'logistic', 'naive_bayes', 'svm')
- `split_data()` - Train/test split
- `train_model()` - Train the classifier
- `save_model()` / `load_model()` - Persist model

### src/evaluate.py
Model evaluation:
- `evaluate_model()` - Complete evaluation with all metrics
- `get_sentiment_statistics()` - Calculate sentiment distribution

### src/visualizations.py
NLP visualizations:
- `create_word_cloud()` - Generate word clouds
- `create_sentiment_word_clouds()` - Create word clouds per sentiment
- `plot_confusion_matrix()` - Plot confusion matrix with annotations
- `plot_feature_importance()` - Plot feature importance
- `plot_sentiment_distribution()` - Plot sentiment distribution
- `plot_text_length_vs_sentiment()` - Plot text length analysis
- `generate_all_visualizations()` - Generate all visualizations at once

### src/error_analysis.py
Error analysis module:
- `analyze_errors()` - Analyze misclassified examples
- `print_error_analysis()` - Print formatted error report
- `identify_confusing_cases()` - Identify ambiguous cases
- `analyze_error_patterns()` - Analyze patterns in errors

### src/text_analysis.py
Text analysis features:
- `extract_keywords()` - Extract top keywords using TF-IDF
- `extract_keywords_by_sentiment()` - Extract keywords per sentiment
- `highlight_sentiment_words()` - Highlight positive/negative words
- `get_sentiment_word_counts()` - Count sentiment words
- `calculate_text_complexity()` - Calculate readability metrics
- `analyze_text()` - Comprehensive text analysis

## Evaluation Metrics

The model is evaluated using:

- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: How many selected items are relevant
- **Recall**: How many relevant items are selected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True vs predicted labels breakdown

## Troubleshooting

### "Model files not found" Error
Run `python main.py` first to train and save the model.

### NLTK Data Download Issues
If NLTK data fails to download automatically, run manually:
```python
import nltk
nltk.download('stopwords')
```

### Port Already in Use
If port 5000 is busy, modify `app.py`:
```python
app.run(port=5001)  # Use different port
```

## Technical Stack

- **Python 3.8+**
- **scikit-learn** - Machine Learning
- **NLTK** - Natural Language Processing
- **Flask** - Web Framework
- **pandas/numpy** - Data Processing
- **joblib** - Model Serialization
- **wordcloud** - Word cloud generation
- **matplotlib** - Plotting and visualizations
- **seaborn** - Enhanced statistical visualizations

## NLP Features

### Advanced Preprocessing
The project uses advanced NLP preprocessing techniques:
- **Lemmatization**: Converts words to their base form using WordNetLemmatizer
- **Stemming**: Alternative method using PorterStemmer for root word extraction
- **Emoji Handling**: Converts emojis to text descriptions (e.g., 😀 → "happy_face")
- **Slang Normalization**: Normalizes common abbreviations (e.g., "u" → "you", "lol" → "laugh_out_loud")
- **Negation Detection**: Marks negated words to preserve context after stopword removal
- **Configurable Pipeline**: All preprocessing steps can be enabled/disabled via configuration

### Feature Extraction
Multiple feature extraction methods for better text representation:
- **N-grams**: Supports unigrams, bigrams, and trigrams
- **Character N-grams**: Captures morphological patterns (3-5 character sequences)
- **Text Features**: Word count, character count, average word length
- **TF-IDF**: Term frequency-inverse document frequency with configurable parameters

### Multiple Classifiers
Support for multiple NLP-specific classifiers:
- **Logistic Regression**: Baseline model with class balancing
- **Naive Bayes (MultinomialNB)**: Classic NLP classifier for text
- **SVM (LinearSVC)**: Effective for high-dimensional text data

### Visualizations
Comprehensive NLP visualizations generated during training:
- **Word Clouds**: Per sentiment class showing most frequent words
- **Confusion Matrix**: Visual representation of prediction errors
- **Feature Importance**: Top features (words) for classification
- **Sentiment Distribution**: Class distribution in dataset
- **Text Length Analysis**: Text length distribution by sentiment

### Error Analysis
Detailed error analysis to understand model weaknesses:
- **Error Types**: Breakdown of misclassification types (e.g., "positive_as_negative")
- **Error Statistics**: Error rate per sentiment class
- **Sample Errors**: Examples of misclassified texts
- **Pattern Analysis**: Common patterns in errors (short text, negation, mixed sentiment)

### Text Analysis
Real-time text analysis features in the web interface:
- **Sentiment Word Highlighting**: Highlights positive (green) and negative (red) words
- **Keyword Extraction**: Top keywords per sentiment class
- **Sentiment Word Counts**: Count of positive/negative words in text
- **Text Complexity**: Readability metrics (Flesch Reading Ease, word/sentence length)

## Future Enhancements

- Implement hyperparameter tuning with GridSearchCV
- Add model comparison dashboard for side-by-side evaluation
- Support for additional languages
- Export results to CSV/Excel
- Add real-time sentiment monitoring for social media feeds

## License

This project is open source and available for educational purposes.

## Author

Created for learning NLP and Machine Learning with Python.

---

**Happy Analyzing!** 🚀
