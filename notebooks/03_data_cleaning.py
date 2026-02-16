import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

# Load data
df = pd.read_csv('data/reviews_raw.csv')

print("\n" + "="*60)
print("CLEANING DATA")
print("="*60)

# Store original size
original_size = len(df)
print(f"Original dataset size: {original_size:,}")

# Step 1: Remove missing reviewText
df_clean = df.dropna(subset=['reviewText']).copy()
print(f"After removing missing reviewText: {len(df_clean):,} ({len(df_clean)/original_size*100:.1f}%)")

# Step 2: Remove empty strings
df_clean = df_clean[df_clean['reviewText'].str.strip() != ''].copy()
print(f"After removing empty reviews: {len(df_clean):,} ({len(df_clean)/original_size*100:.1f}%)")

# Step 3: Recalculate text stats on clean data
df_clean['text_length'] = df_clean['reviewText'].astype(str).apply(len)
df_clean['word_count'] = df_clean['reviewText'].astype(str).apply(lambda x: len(x.split()))

# Step 4: Remove very short reviews
df_clean = df_clean[df_clean['word_count'] >= 3].copy()
print(f"After removing very short reviews: {len(df_clean):,} ({len(df_clean)/original_size*100:.1f}%)")

# # Step 5: Remove very long reviews (potential spam)
# df_clean = df_clean[df_clean['word_count'] <= 500].copy()
# print(f"After removing very long reviews: {len(df_clean):,} ({len(df_clean)/original_size*100:.1f}%)")

# Step 6: Remove duplicates
df_clean = df_clean.drop_duplicates(subset=['reviewText']).copy()
print(f"After removing duplicates: {len(df_clean):,} ({len(df_clean)/original_size*100:.1f}%)")

print(f"\n{'='*60}")
print(f"FINAL CLEAN DATASET: {len(df_clean):,} reviews")
print(f"Removed: {original_size - len(df_clean):,} reviews ({(original_size - len(df_clean))/original_size*100:.1f}%)")
print('='*60)

# Check rating distribution after cleaning
print("\nRating distribution after cleaning:")
print(df_clean['overall'].value_counts().sort_index())

# Visualize before/after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['overall'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='lightcoral', edgecolor='black')
axes[0].set_title('Before Cleaning', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

df_clean['overall'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
axes[1].set_title('After Cleaning', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Save cleaned data
output_file = 'data/reviews_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n{'='*60}")
print(f"Cleaned data saved to: {output_file}")
print(f"Total reviews in cleaned dataset: {len(df_clean):,}")