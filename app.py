"""
Flask Web Application for Sentiment Analysis
============================================
This module provides a web interface for the sentiment analysis model.
Users can input comments and get real-time sentiment predictions.

Usage:
    python app.py
    
Then open your browser and go to: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_text
from features import load_vectorizer
from train import load_model, predict, predict_proba
from evaluate import get_sentiment_statistics
from text_analysis import highlight_sentiment_words, analyze_text

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None


def init_model():
    """
    Load the trained model and vectorizer.
    Called once when the app starts.
    """
    global model, vectorizer
    
    model_path = os.path.join('models', 'sentiment_model.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("=" * 70)
        print("ERROR: Model files not found!")
        print("=" * 70)
        print(f"\nExpected files:")
        print(f"  - {model_path}")
        print(f"  - {vectorizer_path}")
        print(f"\nPlease run 'python main.py' first to train the model.")
        print("=" * 70)
        sys.exit(1)
    
    print("Loading model and vectorizer...")
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    print("Model loaded successfully!")


def analyze_single_comment(comment):
    """
    Analyze a single comment and return sentiment prediction with text analysis.
    
    Args:
        comment (str): Raw comment text
        
    Returns:
        dict: Prediction results including sentiment, confidence, and text analysis
    """
    # Preprocess the comment
    processed_comment = preprocess_text(comment)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([processed_comment])
    
    # Get prediction
    sentiment = predict(model, features)[0]
    
    # Get prediction probabilities
    probabilities = predict_proba(model, features)[0]
    
    # Get class labels
    classes = model.classes_
    
    # Create confidence dictionary
    confidence = {
        cls: round(prob * 100, 2) 
        for cls, prob in zip(classes, probabilities)
    }
    
    # Get confidence for predicted sentiment
    predicted_confidence = confidence[sentiment]
    
    # Perform text analysis
    text_analysis = analyze_text(comment)
    
    # Highlight sentiment words
    highlighted_text = highlight_sentiment_words(comment)
    
    return {
        'original_text': comment,
        'processed_text': processed_comment,
        'sentiment': sentiment,
        'confidence': predicted_confidence,
        'all_confidences': confidence,
        'highlighted_text': highlighted_text,
        'text_analysis': text_analysis
    }


def analyze_multiple_comments(comments):
    """
    Analyze multiple comments and return results with statistics.
    
    Args:
        comments (list): List of comment strings
        
    Returns:
        dict: Analysis results including individual predictions and statistics
    """
    results = []
    
    for comment in comments:
        if comment.strip():  # Skip empty lines
            result = analyze_single_comment(comment)
            results.append(result)
    
    # Calculate statistics
    predictions = [r['sentiment'] for r in results]
    stats = get_sentiment_statistics(predictions)
    
    return {
        'predictions': results,
        'statistics': stats
    }


@app.route('/')
def index():
    """
    Render the home page with sentiment analyzer.
    """
    return render_template('index.html', active_page='home')


@app.route('/analytics')
def analytics():
    """
    Render the analytics page with statistics and charts.
    """
    return render_template('analytics.html', active_page='analytics')


@app.route('/about')
def about():
    """
    Render the about page with project information.
    """
    return render_template('about.html', active_page='about')


@app.route('/analysis')
def analysis():
    """
    Render the NLP analysis dashboard with visualizations and insights.
    """
    return render_template('analysis.html', active_page='analysis')


@app.route('/models/visualizations/<path:filename>')
def serve_visualization(filename):
    """
    Serve visualization files from the models/visualizations directory.
    """
    from flask import send_from_directory
    return send_from_directory('models/visualizations', filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle POST request for sentiment analysis.
    Accepts both single comment and multiple comments.
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text_input = data['text']
    
    # Check if input is a single comment or multiple (separated by newlines)
    comments = [line.strip() for line in text_input.split('\n') if line.strip()]
    
    if len(comments) == 1:
        # Single comment
        result = analyze_single_comment(comments[0])
        return jsonify(result)
    else:
        # Multiple comments
        result = analyze_multiple_comments(comments)
        return jsonify(result)


@app.route('/api/stats')
def api_stats():
    """
    Return dataset statistics for analytics page.
    """
    import pandas as pd
    
    try:
        df = pd.read_csv('data/raw/comments.csv')
        counts = df['label'].value_counts()
        
        return jsonify({
            'total': len(df),
            'positive': int(counts.get('positive', 0)),
            'negative': int(counts.get('negative', 0)),
            'neutral': int(counts.get('neutral', 0)),
            'accuracy': 98.14,
            'model': 'Logistic Regression',
            'features': 'TF-IDF (5000)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def api_info():
    """
    Return API information and available endpoints.
    """
    return jsonify({
        'name': 'Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            '/': 'Home - Sentiment Analyzer (GET)',
            '/analytics': 'Analytics Dashboard (GET)',
            '/about': 'About Project (GET)',
            '/analyze': 'Analyze sentiment (POST)',
            '/api/stats': 'Dataset statistics (GET)',
            '/api/info': 'API information (GET)'
        },
        'example_request': {
            'text': 'This product is amazing! I love it.'
        }
    })


if __name__ == '__main__':
    # Initialize model before starting server
    init_model()
    
    print("\n" + "=" * 70)
    print("Starting Flask server...")
    print("=" * 70)
    print("Open your browser and go to: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
