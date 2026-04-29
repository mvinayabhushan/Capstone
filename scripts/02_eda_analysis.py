# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 02: Exploratory Data Analysis (EDA)
# Author: Vinaya Bhushan M | Walsh College
# ============================================================
# Outputs:
#   outputs/eda/descriptive_stats.csv
#   outputs/eda/correlation_matrix.csv
#   outputs/eda/fig_1_target_distribution.png
#   outputs/eda/fig_2_age_distribution_by_chd.png
#   outputs/eda/fig_3_max_hr_vs_age.png
#   outputs/eda/fig_4_correlation_heatmap.png
#   outputs/eda/fig_5_cholesterol_by_chd.png
#   outputs/eda/fig_6_st_depression_by_chd.png
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = 'data/heart_disease_cleaned.csv'
OUT_DIR = 'outputs/eda'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS — CHD Dataset")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")

# ── 1. Descriptive Statistics (printed + saved) ──────────────
desc = df.describe().round(2)
print(f"\nDescriptive Statistics:\n{desc}")
desc.to_csv(f'{OUT_DIR}/descriptive_stats.csv')
print(f"\nSaved: {OUT_DIR}/descriptive_stats.csv")

# ── 2. Target Distribution ───────────────────────────────────
vc = df['target_binary'].value_counts()
print(f"\nTarget Distribution:")
print(f"  No Disease (0): {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"  Disease    (1): {vc[1]} ({vc[1]/len(df)*100:.1f}%)")

# ── 3. Gender Distribution ───────────────────────────────────
print(f"\nGender Distribution:")
n_male = (df['sex'] == 1).sum()
n_female = (df['sex'] == 0).sum()
print(f"  Male   (1): {n_male} ({n_male/len(df)*100:.1f}%)")
print(f"  Female (0): {n_female} ({n_female/len(df)*100:.1f}%)")
print("  NOTE: Dataset is ~79% male — documented as bias limitation in synopsis.")

# ── 4. Key Feature Means by Disease Status ───────────────────
print(f"\nKey Feature Comparison by Disease Status:")
for col in ['age', 'thalach', 'oldpeak', 'trestbps', 'chol']:
    no_d = df[df['target_binary'] == 0][col].mean()
    yes_d = df[df['target_binary'] == 1][col].mean()
    print(f"  {col:10s} | No Disease: {no_d:6.1f} | Disease: {yes_d:6.1f}")

# ── 5. Correlation with Target ───────────────────────────────
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
corrs = df[numeric_cols + ['target_binary']].corr()['target_binary'].drop('target_binary')
print(f"\nCorrelation with Target Variable:")
for feat, corr in corrs.sort_values(key=abs, ascending=False).items():
    print(f"  {feat}: {corr:+.3f}")

# Save full correlation matrix
corr_full = df[numeric_cols + ['target_binary']].corr().round(3)
corr_full.to_csv(f'{OUT_DIR}/correlation_matrix.csv')
print(f"Saved: {OUT_DIR}/correlation_matrix.csv")

# ── 6. Disease Rate by Chest Pain Type ───────────────────────
print(f"\nDisease Rate by Chest Pain Type:")
cp_map = {1: 'Typical Angina', 2: 'Atypical Angina',
          3: 'Non-anginal Pain', 4: 'Asymptomatic'}
for cp_code, name in cp_map.items():
    subset = df[df['cp'] == cp_code]
    if len(subset) > 0:
        rate = subset['target_binary'].mean() * 100
        print(f"  {name} (n={len(subset)}): {rate:.1f}%")

# ── 7. Visualisations — one chart per file ───────────────────
sns.set_style('whitegrid')
COLOR_NO = '#2ecc71'
COLOR_YES = '#e74c3c'

# Figure 1 — Target Distribution
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['target_binary'].value_counts().sort_index()
ax.bar(['No Disease', 'Disease'], counts, color=[COLOR_NO, COLOR_YES])
for i, v in enumerate(counts):
    ax.text(i, v + 5, f"{v}\n({v/len(df)*100:.1f}%)",
            ha='center', fontweight='bold')
ax.set_title('Figure 1. Target Distribution (n=918)', fontweight='bold')
ax.set_ylabel('Number of Patients')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_1_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_1_target_distribution.png")

# Figure 2 — Age Distribution by CHD Status
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[df['target_binary'] == 0]['age'], bins=15, alpha=0.6,
        color=COLOR_NO, label='No Disease', edgecolor='black')
ax.hist(df[df['target_binary'] == 1]['age'], bins=15, alpha=0.6,
        color=COLOR_YES, label='Disease', edgecolor='black')
ax.set_title('Figure 2. Age Distribution by CHD Status', fontweight='bold')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of Patients')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_2_age_distribution_by_chd.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_2_age_distribution_by_chd.png")

# Figure 3 — Max HR vs Age (scatter)
fig, ax = plt.subplots(figsize=(8, 5))
for val, color, label in [(0, COLOR_NO, 'No Disease'), (1, COLOR_YES, 'Disease')]:
    subset = df[df['target_binary'] == val]
    ax.scatter(subset['age'], subset['thalach'],
               alpha=0.5, color=color, label=label, s=25, edgecolor='white', linewidth=0.5)
ax.set_title('Figure 3. Maximum Heart Rate vs Age, by CHD Status', fontweight='bold')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Maximum Heart Rate (bpm)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_3_max_hr_vs_age.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_3_max_hr_vs_age.png")

# Figure 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_full, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=-1, vmax=1, center=0,
            annot_kws={'size': 10})
ax.set_title('Figure 4. Correlation Matrix — Numeric Features and Target', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_4_correlation_heatmap.png")

# Figure 5 — Cholesterol by CHD Status (seaborn boxplot)
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='target_binary', y='chol',
            palette=[COLOR_NO, COLOR_YES], ax=ax)
ax.set_title('Figure 5. Cholesterol by CHD Status', fontweight='bold')
ax.set_xlabel('CHD Status')
ax.set_ylabel('Serum Cholesterol (mg/dl)')
ax.set_xticklabels(['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_5_cholesterol_by_chd.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_5_cholesterol_by_chd.png")

# Figure 6 — ST Depression (oldpeak) by CHD Status
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='target_binary', y='oldpeak',
            palette=[COLOR_NO, COLOR_YES], ax=ax)
ax.set_title('Figure 6. ST Depression (Oldpeak) by CHD Status', fontweight='bold')
ax.set_xlabel('CHD Status')
ax.set_ylabel('ST Depression (exercise vs rest)')
ax.set_xticklabels(['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_6_st_depression_by_chd.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_6_st_depression_by_chd.png")

print("\n" + "=" * 60)
print(f"EDA complete. All outputs in: {OUT_DIR}/")
print("=" * 60)
