import os

files = os.listdir('data/raw')
print('Files in data/raw:')
for f in files:
    size_mb = os.path.getsize(os.path.join('data/raw', f)) / (1024*1024)
    print(f'  {f}: {size_mb:.2f} MB')
