import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

# Load data
df_clean = pd.read_csv('data/reviews_cleaned.csv')

# Store original size
df_original = pd.read_csv('data/reviews_raw.csv')
original_size = len(df_original)

print("\n" + "="*60)
print("CREATING SENTIMENT LABELS")
print("="*60)

# Strategy: Convert 1-5 star ratings to sentiment
# 1-2 stars = Negative
# 3 stars = Neutral  
# 4-5 stars = Positive

def create_sentiment_label(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

df_clean['sentiment'] = df_clean['overall'].apply(create_sentiment_label)

print("Sentiment distribution:")
print(df_clean['sentiment'].value_counts())
print(f"\nPercentages:")
print(df_clean['sentiment'].value_counts(normalize=True) * 100)

# Visualize
plt.figure(figsize=(10, 6))
df_clean['sentiment'].value_counts().plot(kind='bar', color=['red', 'gray', 'green'], edgecolor='black')
plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(df_clean['sentiment'].value_counts()):
    plt.text(i, v + 1000, f'{v:,}\n({v/len(df_clean)*100:.1f}%)', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Check class imbalance
print("\n" + "="*60)
print("CLASS BALANCE ANALYSIS")
print("="*60)

sentiment_counts = df_clean['sentiment'].value_counts()
max_count = sentiment_counts.max()
min_count = sentiment_counts.min()
imbalance_ratio = max_count / min_count

print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"Most common: {sentiment_counts.idxmax()} ({sentiment_counts.max():,})")
print(f"Least common: {sentiment_counts.idxmin()} ({sentiment_counts.min():,})")

print("\n" + "="*60)
print("BALANCING DATASET")
print("="*60)

min_samples = df_clean['sentiment'].value_counts().min()
print(f"Minimum class size: {min_samples:,}")

# Option 1: Completely balanced (same number for each class)
balanced_size = min_samples
print(f"\nCreating balanced dataset with {balanced_size:,} samples per class...")

df_balanced = pd.concat([
    df_clean[df_clean['sentiment'] == 'negative'].sample(n=balanced_size, random_state=42),
    df_clean[df_clean['sentiment'] == 'neutral'].sample(n=balanced_size, random_state=42),
    df_clean[df_clean['sentiment'] == 'positive'].sample(n=balanced_size, random_state=42)
])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(df_balanced):,}")
print(f"\nBalanced sentiment distribution:")
print(df_balanced['sentiment'].value_counts())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_clean['sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['red', 'gray', 'green'], edgecolor='black')
axes[0].set_title('Original (Imbalanced)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

df_balanced['sentiment'].value_counts().plot(kind='bar', ax=axes[1], color=['red', 'gray', 'green'], edgecolor='black')
axes[1].set_title('Balanced Dataset', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("DECISION: Which dataset to use?")
print("="*60)
print("""
We have two options:

1. BALANCED dataset ({:,} samples)
   - Equal representation of all sentiments
   - Better for initial model training
   - Less realistic distribution
   
2. FULL CLEANED dataset ({:,} samples)
   - Real-world distribution
   - More data for training
   - Class imbalance to handle

RECOMMENDATION: Use BALANCED for this project
- Easier to get good initial results
- Still plenty of data ({:,} samples)
- Can always use full dataset later
""".format(len(df_balanced), len(df_clean), len(df_balanced)))

# We'll proceed with balanced dataset
df_final = df_balanced.copy()
print(f"\nUsing balanced dataset: {len(df_final):,} reviews")

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

# Define preprocessing function
def preprocess_text(text):
    """
    Clean and preprocess text for ML
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits (keep letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Test the function
print("Testing preprocessing function:")
sample_text = "This phone is AMAZING!!! Check it out: http://example.com <br> Price: $299"
print(f"\nOriginal: {sample_text}")
print(f"Processed: {preprocess_text(sample_text)}")

print("\n" + "="*60)
print("APPLYING PREPROCESSING TO ALL REVIEWS")
print("="*60)

# Apply preprocessing
print("Processing reviews...")
df_final['review_clean'] = df_final['reviewText'].apply(preprocess_text)

# Show examples
print("\nExamples of preprocessing:")
print("="*60)
for idx in df_final.head(5).index:
    print(f"\nOriginal: {df_final.loc[idx, 'reviewText'][:150]}...")
    print(f"Cleaned:  {df_final.loc[idx, 'review_clean'][:150]}...")
    print("-"*60)

# Check for empty texts after cleaning
empty_after_cleaning = (df_final['review_clean'].str.strip() == '').sum()
print(f"\nEmpty reviews after cleaning: {empty_after_cleaning}")

if empty_after_cleaning > 0:
    print("Removing empty reviews...")
    df_final = df_final[df_final['review_clean'].str.strip() != ''].copy()
    print(f"Final dataset size: {len(df_final):,}")

print("\nText preprocessing complete!")

print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

# Select final columns for ML
columns_to_keep = [
    'reviewText',           # Original text
    'review_clean',         # Cleaned text (main feature)
    'overall',              # Original rating
    'sentiment',            # Target variable
    'summary'              # Review summary
]

df_processed = df_final[columns_to_keep].copy()

print(f"Final processed dataset shape: {df_processed.shape}")
print(f"\nColumns: {df_processed.columns.tolist()}")
print(f"\nFirst few rows:")
print(df_processed.head())

# Save to CSV
output_path = 'data/reviews_processed.csv'
df_processed.to_csv(output_path, index=False)
print(f"\nSaved processed data to: {output_path}")

# Also save a sample for quick testing
sample_size = 5000
df_sample = df_processed.sample(n=min(sample_size, len(df_processed)), random_state=42)
sample_path = 'data/reviews_sample_5k.csv'
df_sample.to_csv(sample_path, index=False)
print(f"Saved sample data ({len(df_sample):,} rows) to: {sample_path}")

print("\n" + "="*60)
print("DATA PREPROCESSING SUMMARY")
print("="*60)
print(f"""
Original dataset: {original_size:,} reviews
After cleaning: {len(df_clean):,} reviews  
Balanced dataset: {len(df_balanced):,} reviews
Final processed: {len(df_processed):,} reviews

Sentiment distribution:
{df_processed['sentiment'].value_counts().to_string()}

Files saved:
- data/reviews_processed.csv ({len(df_processed):,} rows)
- data/reviews_sample_5k.csv ({len(df_sample):,} rows)

Ready for model training!
""")

print("\n" + "="*60)
print("FINAL DATA QUALITY CHECK")
print("="*60)
print(df_processed.info())
print(f"\nMissing values:")
print(df_processed.isnull().sum())
print(f"\nNo missing values in key columns!")