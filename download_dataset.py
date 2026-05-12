"""
Kaggle Dataset Downloader
Downloads sentiment analysis dataset from Kaggle
"""
import os
import sys

# Set Kaggle API token
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_a704fe0be7c4705119ad8c82669d2af6'
os.environ['KAGGLE_USERNAME'] = 'user'

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("=" * 60)
    print("KAGGLE DATASET DOWNLOAD")
    print("=" * 60)
    
    # Initialize API
    api = KaggleApi()
    print("Authenticating with Kaggle...")
    
    # Try to authenticate
    try:
        api._load_config()
        print(f"✓ Config loaded")
        print(f"✓ Token exists: {bool(api.api_token)}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Search for sentiment datasets
    print("\nSearching for sentiment analysis datasets...")
    datasets = api.dataset_list(search='sentiment', file_type='csv', page=1)
    
    print(f"\nFound {len(datasets)} datasets:")
    print("-" * 60)
    
    for i, d in enumerate(datasets[:10]):
        print(f"{i+1}. {d.ref}")
        print(f"   Title: {d.title}")
        print(f"   Size: {getattr(d, 'size', 'Unknown')}")
        print(f"   Downloads: {getattr(d, 'downloadCount', 'Unknown')}")
        print()
    
    # Find a good 3-class sentiment dataset
    target_dataset = None
    for d in datasets:
        title_lower = d.title.lower()
        if any(x in title_lower for x in ['sentiment', 'emotion', 'twitter']):
            if '3' in title_lower or 'three' in title_lower or 'positive' in title_lower:
                target_dataset = d.ref
                print(f"\n✓ Selected dataset: {d.ref}")
                print(f"  Title: {d.title}")
                break
    
    if not target_dataset and datasets:
        target_dataset = datasets[0].ref
        print(f"\n✓ Selected first dataset: {target_dataset}")
    
    if target_dataset:
        # Download dataset
        raw_dir = os.path.join('data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        print(f"\nDownloading to: {raw_dir}")
        api.dataset_download_files(target_dataset, path=raw_dir, unzip=True)
        print("✓ Download complete!")
        
        # List downloaded files
        files = os.listdir(raw_dir)
        print(f"\nDownloaded files:")
        for f in files:
            print(f"  - {f}")
        
        sys.exit(0)
    else:
        print("No suitable dataset found!")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
