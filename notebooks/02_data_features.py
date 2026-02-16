import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

# Load data
df = pd.read_csv('data/reviews_raw.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Total reviews: {len(df):,}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n" + "="*60)
print("RATING DISTRIBUTION")
print("="*60)
print(df['overall'].value_counts().sort_index())
print(f"\nMean rating: {df['overall'].mean():.2f}")
print(f"Median rating: {df['overall'].median():.2f}")

# Visualize rating distribution
plt.figure(figsize=(10, 6))
df['overall'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Rating Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(df['overall'].value_counts().sort_index()):
    plt.text(i, v + 1000, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("TEXT ANALYSIS")
print("="*60)

# Check for missing review text
print(f"Missing reviewText: {df['reviewText'].isnull().sum()}")
print(f"Empty reviewText: {(df['reviewText'] == '').sum()}")

# Remove missing reviews
print(f"\nRemoving {df['reviewText'].isnull().sum()} rows with missing reviewText...")
df = df.dropna(subset=['reviewText'])
print(f"Remaining reviews: {len(df):,}")

# Calculate text statistics
df['text_length'] = df['reviewText'].astype(str).apply(len)
df['word_count'] = df['reviewText'].astype(str).apply(lambda x: len(x.split()))

print(f"\nText Statistics:")
print(f"Average characters: {df['text_length'].mean():.0f}")
print(f"Average words: {df['word_count'].mean():.0f}")
print(f"\nText length distribution:")
print(df['text_length'].describe())

# Visualize text lengths
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(df['text_length'], bins=50, color='coral', edgecolor='black')
axes[0].set_title('Review Length (Characters)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['text_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["text_length"].mean():.0f}')
axes[0].legend()

axes[1].hist(df['word_count'], bins=50, color='lightgreen', edgecolor='black')
axes[1].set_title('Review Length (Words)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["word_count"].mean():.0f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("SAMPLE REVIEWS BY RATING")
print("="*60)

for rating in [5, 3, 1]:
    print(f"\n{'='*60}")
    print(f"RATING {rating} - Examples")
    print('='*60)
    samples = df[df['overall'] == rating].head(3)
    for idx, row in samples.iterrows():
        print(f"\nSummary: {row['summary']}")
        print(f"Review: {row['reviewText'][:300]}...")
        print(f"Length: {row['word_count']} words")

print("\n" + "="*60)
print("DATA QUALITY ISSUES TO HANDLE")
print("="*60)

# Check for duplicates
duplicates = df.duplicated(subset=['reviewText']).sum()
print(f"1. Duplicate reviews: {duplicates:,}")

# Check for very short reviews
very_short = (df['word_count'] < 3).sum()
print(f"2. Very short reviews (<3 words): {very_short:,}")

# Check for very long reviews (might be spam)
very_long = (df['word_count'] > 500).sum()
print(f"3. Very long reviews (>500 words): {very_long:,}")