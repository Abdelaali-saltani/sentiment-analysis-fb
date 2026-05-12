"""
Process the large sentiment_analysis.csv dataset
"""
import os
import pandas as pd
import sys

print("=" * 70)
print("PROCESSING LARGE DATASET")
print("=" * 70)

try:
    # Read in chunks to handle large file
    chunk_size = 10000
    chunks = []
    
    print("\nReading sentiment_analysis.csv in chunks...")
    chunk_count = 0
    
    for chunk in pd.read_csv('data/raw/sentiment_analysis.csv', usecols=['text', 'sentiment'], chunksize=chunk_size):
        chunks.append(chunk)
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count * chunk_size:,} rows...")
    
    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)
    
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
        print(f"\n✓ Dataset has {len(unique_sentiments)} classes")
        
        # Check if labels need mapping
        required_labels = {'positive', 'negative', 'neutral'}
        current_labels = set(df['sentiment'].unique())
        
        # Normalize labels
        label_mapping = {}
        for label in current_labels:
            label_str = str(label).lower().strip()
            if label_str in ['positive', 'pos', '1'] or 'pos' in label_str:
                label_mapping[label] = 'positive'
            elif label_str in ['negative', 'neg', '-1'] or 'neg' in label_str:
                label_mapping[label] = 'negative'
            elif label_str in ['neutral', 'neu', '0'] or 'neu' in label_str:
                label_mapping[label] = 'neutral'
            else:
                label_mapping[label] = label_str
        
        print(f"\nLabel mapping: {label_mapping}")
        df['sentiment'] = df['sentiment'].map(label_mapping)
        
        # Filter to only keep valid labels
        valid_labels = ['positive', 'negative', 'neutral']
        df = df[df['sentiment'].isin(valid_labels)]
        
        # Rename columns
        df = df.rename(columns={'text': 'comment', 'sentiment': 'label'})
        
        # Remove empty comments
        df = df.dropna(subset=['comment'])
        df = df[df['comment'].str.strip() != '']
        
        print(f"\n✓ Final dataset: {len(df):,} samples")
        
        # Save
        output_path = 'data/raw/comments.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to: {output_path}")
        
        # Print statistics
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
        
        # Verify minimum requirement
        if len(df) >= 3000:
            print(f"\n✓ Dataset meets requirement: {len(df):,} >= 3,000 samples")
        else:
            print(f"\n⚠ Dataset has {len(df):,} samples (need 3,000+)")
        
        print("\n✓ Dataset ready for training!")
        sys.exit(0)
    else:
        print(f"\n✗ Only {len(unique_sentiments)} classes found")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
