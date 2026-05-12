"""
Process downloaded Kaggle dataset for sentiment analysis
Converts to standard format: comment,label with 3 classes
"""
import os
import pandas as pd
import sys

print("=" * 70)
print("DATASET PROCESSING")
print("=" * 70)

# Check which files were downloaded
raw_dir = 'data/raw'
files = os.listdir(raw_dir)
print(f"\nFiles in {raw_dir}:")
for f in files:
    size = os.path.getsize(os.path.join(raw_dir, f)) / (1024*1024)
    print(f"  - {f} ({size:.2f} MB)")

# Try to read and process sentiment_analysis.csv
print("\n" + "-" * 70)
print("Processing sentiment_analysis.csv...")
print("-" * 70)

try:
    # Read only necessary columns
    df = pd.read_csv('data/raw/sentiment_analysis.csv', usecols=['text', 'sentiment'])
    
    print(f"\n✓ Loaded {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check unique sentiments
    unique_sentiments = df['sentiment'].unique()
    print(f"\nUnique sentiments: {list(unique_sentiments)}")
    
    # Check sentiment distribution
    print("\nSentiment distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / len(df) * 100
        print(f"  {sentiment}: {count:,} ({pct:.1f}%)")
    
    # Check if we have 3 classes
    if len(unique_sentiments) >= 3:
        print(f"\n✓ Dataset has {len(unique_sentiments)} classes - Good!")
        
        # Check if we need to map numeric labels
        if all(isinstance(x, (int, float)) or str(x).isdigit() for x in unique_sentiments[:3]):
            print("\nNumeric labels detected. Mapping to text labels...")
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df['sentiment'] = df['sentiment'].map(label_map)
            print("Labels mapped!")
        
        # Check if we have required labels
        required_labels = {'positive', 'negative', 'neutral'}
        current_labels = set(df['sentiment'].unique())
        
        if required_labels.issubset(current_labels):
            print("\n✓ All required labels (positive, negative, neutral) present!")
        else:
            print(f"\n⚠ Missing some labels. Have: {current_labels}")
            print(f"  Need: {required_labels}")
        
        # If more than 3000 samples, sample to get a good distribution
        if len(df) > 3000:
            print(f"\nDataset has {len(df):,} samples (requirement met: 3000+)")
            
            # Keep all data or sample if too large (more than 100k)
            if len(df) > 100000:
                print("Dataset very large - sampling 50,000 for manageable training...")
                # Sample while maintaining class distribution
                df = df.groupby('sentiment').apply(
                    lambda x: x.sample(n=min(len(x), 17000), random_state=42)
                ).reset_index(drop=True)
                print(f"Sampled to {len(df):,} samples")
        
        # Rename columns to standard format
        df = df.rename(columns={'text': 'comment', 'sentiment': 'label'})
        
        # Ensure consistent label naming
        label_mapping = {
            'positive': 'positive',
            'negative': 'negative', 
            'neutral': 'neutral',
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral'
        }
        df['label'] = df['label'].str.lower().map(lambda x: label_mapping.get(x, x))
        
        # Remove any empty comments
        df = df.dropna(subset=['comment'])
        df = df[df['comment'].str.strip() != '']
        
        # Save to comments.csv
        output_path = 'data/raw/comments.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed dataset to: {output_path}")
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("FINAL DATASET STATISTICS")
        print("=" * 70)
        print(f"\nTotal samples: {len(df):,}")
        print("\nClass distribution:")
        final_counts = df['label'].value_counts()
        for label in ['positive', 'negative', 'neutral']:
            if label in final_counts:
                count = final_counts[label]
                pct = count / len(df) * 100
                print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        print("\n✓ Dataset ready for training!")
        sys.exit(0)
    else:
        print(f"\n✗ Dataset only has {len(unique_sentiments)} classes, need at least 3")
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ Error processing dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
