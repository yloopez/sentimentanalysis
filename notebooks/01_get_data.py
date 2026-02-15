import pandas as pd
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Create data directory
os.makedirs('data', exist_ok=True)

print("=" * 60)
print("DOWNLOADING AMAZON REVIEWS DATASET FROM KAGGLE")
print("=" * 60)

# Initialize and authenticate Kaggle API
print("\n[1/4] Authenticating with Kaggle API...")
api = KaggleApi()
api.authenticate()
print("Authentication successful!")

# Download the dataset
print("\n[2/4] Downloading dataset (this may take a few minutes)...")
try:
    api.dataset_download_files(
        'abdallahwagih/amazon-reviews',
        path='data/',
        unzip=True,
        quiet=False
    )
    print("Download and extraction complete!")
except Exception as e:
    print(f"Error downloading: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure kaggle.json is in C:\\Users\\User\\.kaggle\\")
    print("2. Check your internet connection")
    print("3. Verify the dataset exists at the URL")
    exit()

# Find downloaded files
print("\n[3/4] Looking for downloaded files...")
files = os.listdir('data/')
print(f"Files found: {files}")

# Look for JSON or CSV files
json_files = [f for f in files if f.endswith('.json')]
csv_files = [f for f in files if f.endswith('.csv')]

# Load the data
print("\n[4/4] Loading data into DataFrame...")

if csv_files:
    data_file = csv_files[0]
    print(f"Loading CSV file: {data_file}")
    df = pd.read_csv(f'data/{data_file}')
    
elif json_files:
    data_file = json_files[0]
    print(f"Loading JSON file: {data_file}")
    
    # Try standard JSON first
    try:
        with open(f'data/{data_file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If it's a dict, it might contain the data under a key
            if len(data) > 0:
                first_key = list(data.keys())[0]
                if isinstance(data[first_key], list):
                    df = pd.DataFrame(data[first_key])
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
        else:
            print(f"Unexpected JSON structure: {type(data)}")
            exit()
    except json.JSONDecodeError:
        # Try JSONL format (line-delimited JSON)
        print("Trying JSONL format (line-delimited)...")
        reviews = []
        with open(f'data/{data_file}', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    reviews.append(json.loads(line))
                    if (i + 1) % 10000 == 0:
                        print(f"  Loaded {i + 1:,} reviews...")
                except:
                    continue
        df = pd.DataFrame(reviews)
else:
    print("No CSV or JSON file found in data/ folder!")
    print(f"Available files: {files}")
    exit()

# Display dataset information
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("None")

# Check for rating/sentiment columns
print("\n" + "-" * 60)
print("KEY COLUMNS ANALYSIS")
print("-" * 60)
for col in df.columns:
    if any(word in col.lower() for word in ['rating', 'score', 'sentiment', 'star']):
        print(f"\n{col}:")
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Distribution:")
            print(df[col].value_counts().sort_index())
        else:
            print(f"  Type: {df[col].dtype}")
            print(f"  Sample values: {df[col].head(3).tolist()}")

# Check text columns
text_cols = [col for col in df.columns if any(word in col.lower() for word in ['text', 'review', 'comment', 'body'])]
if text_cols:
    print(f"\nText columns found: {text_cols}")
    for col in text_cols[:2]:  # Show first 2 text columns
        print(f"\n{col} - Sample:")
        if len(df) > 0:
            sample_text = str(df[col].iloc[0])
            print(f"  {sample_text[:200]}...")

# Save as standardized CSV
output_file = 'data/reviews_raw.csv'
df.to_csv(output_file, index=False)
print("\n" + "=" * 60)
print(f"Data saved to: {output_file}")
print(f"Total reviews: {len(df):,}")
print("=" * 60)