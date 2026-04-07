# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 01: Data Loading & Cleaning
# Author: Vinaya Bhushan M | Walsh College
# ============================================================

import pandas as pd
import numpy as np

# ── Column names (from UCI heart-disease.names) ──────────────
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'target']

# ── Load all 4 datasets ──────────────────────────────────────
base = 'data/'   # adjust path if needed
cleveland    = pd.read_csv(base + 'processed.cleveland.data',  names=cols, na_values='?')
hungarian    = pd.read_csv(base + 'processed.hungarian.data',  names=cols, na_values='?')
switzerland  = pd.read_csv(base + 'processed.switzerland.data',names=cols, na_values='?')
va           = pd.read_csv(base + 'processed.va.data',         names=cols, na_values='?')

df = pd.concat([cleveland, hungarian, switzerland, va], ignore_index=True)
print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Step 1: Remove duplicates ────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {before - len(df)}")

# ── Step 2: Treat zeros as missing for clinical features ─────
df.loc[df['chol'] == 0,     'chol']     = np.nan
df.loc[df['trestbps'] == 0, 'trestbps'] = np.nan

# ── Step 3: Impute low-missing columns with median ───────────
low_missing = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
for col in low_missing:
    median_val = df[col].median()
    missing_count = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: {missing_count} missing values imputed with median ({median_val})")

# ── Step 4: Drop high-missing columns (>33% missing) ─────────
high_missing = ['slope', 'ca', 'thal']
df = df.drop(columns=high_missing)
print(f"Dropped high-missing columns: {high_missing}")

# ── Step 5: Binarise target variable ─────────────────────────
# Original: 0=No Disease, 1-4=Disease severity
# Binarised: 0=No Disease, 1=Disease (any severity)
df['target_binary'] = (df['target'] > 0).astype(int)
print(f"\nTarget distribution:\n{df['target_binary'].value_counts()}")

# ── Step 6: Final check ───────────────────────────────────────
print(f"\nFinal dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# ── Save cleaned dataset ──────────────────────────────────────
df.to_csv('data/heart_disease_cleaned.csv', index=False)
print("\nCleaned dataset saved to: data/heart_disease_cleaned.csv")
