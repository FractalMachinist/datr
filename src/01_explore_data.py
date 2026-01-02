"""
Stage 1: Explore OkCupid dataset
- Load the CSV
- Understand structure, data types, missing values
- Survey traits and essays
"""

import pandas as pd
import numpy as np

# Load dataset
csv_path = '/home/zach/Documents/Proj/datr/downloads/okcupid_profiles.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("DATASET SHAPE & BASIC INFO")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}\n")

print("=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
print(missing_df)

print("\n" + "=" * 80)
print("TRAITS OVERVIEW (Non-Essay Columns)")
print("=" * 80)

# Identify essay columns
essay_cols = [col for col in df.columns if col.startswith('essay')]
trait_cols = [col for col in df.columns if not col.startswith('essay')]

print(f"Essay columns ({len(essay_cols)}): {essay_cols}")
print(f"Trait columns ({len(trait_cols)}): {trait_cols}\n")

# Sample a few rows to see the data
print("First row of traits:")
for col in trait_cols:
    print(f"  {col}: {df[col].iloc[0]}")

print("\n" + "=" * 80)
print("TRAIT VALUE DISTRIBUTIONS")
print("=" * 80)

for col in trait_cols:
    unique_count = df[col].nunique()
    null_count = df[col].isnull().sum()
    print(f"\n{col}:")
    print(f"  Unique values: {unique_count}, Nulls: {null_count}")
    if unique_count <= 15:
        print(f"  Values: {df[col].value_counts().to_dict()}")
    else:
        print(f"  Sample values: {df[col].dropna().unique()[:10]}")

print("\n" + "=" * 80)
print("ESSAY STATISTICS")
print("=" * 80)

for col in essay_cols:
    non_null = df[col].notna().sum()
    avg_len = df[col].dropna().str.len().mean()
    print(f"{col}: {non_null} non-null essays, avg length: {avg_len:.0f} chars")

print("\n" + "=" * 80)
print("SAMPLE ESSAYS")
print("=" * 80)
idx = df[df['essay0'].notna()].index[0]
print(f"\nProfile {idx} - essay0 (My self summary):")
print(df.loc[idx, 'essay0'][:300] + "...")
