import pandas as pd

df = pd.read_csv('data/raw/comments.csv')

print('=' * 70)
print('FINAL DATASET VERIFICATION')
print('=' * 70)

print(f'\nFile: data/raw/comments.csv')
print(f'Total samples: {len(df):,}')
print(f'Columns: {list(df.columns)}')

print('\nFirst 5 rows:')
print(df.head())

print(f'\nUnique labels: {list(df["label"].unique())}')

print('\nClass distribution:')
counts = df['label'].value_counts()
for label in ['positive', 'negative', 'neutral']:
    if label in counts.index:
        count = counts[label]
        pct = count / len(df) * 100
        print(f'  {label}: {count:,} ({pct:.1f}%)')

print('\n' + '=' * 70)
if len(df) >= 3000:
    print('✓ REQUIREMENT MET: 3000+ samples with 3 classes')
else:
    print(f'✗ Need more samples: {len(df)} < 3000')
print('=' * 70)
