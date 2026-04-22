# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 02: Exploratory Data Analysis (EDA)
# Author: Vinaya Bhushan M | Walsh College
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/heart_disease_cleaned.csv')

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS — CHD Dataset")
print("=" * 55)

# ── 1. Basic Info ────────────────────────────────────────────
print(f"\nDataset shape: {df.shape}")
print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")

# ── 2. Target Distribution ───────────────────────────────────
print(f"\nTarget Distribution:")
vc = df['target_binary'].value_counts()
print(f"  No Disease (0): {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"  Disease    (1): {vc[1]} ({vc[1]/len(df)*100:.1f}%)")

# ── 3. Gender Distribution ───────────────────────────────────
print(f"\nGender Distribution:")
print(f"  Male   (1): {(df['sex']==1).sum()} ({(df['sex']==1).sum()/len(df)*100:.1f}%)")
print(f"  Female (0): {(df['sex']==0).sum()} ({(df['sex']==0).sum()/len(df)*100:.1f}%)")
print("  NOTE: Dataset is 79% male — document as bias limitation")

# ── 4. Key Feature Means by Disease Status ───────────────────
print(f"\nKey Feature Comparison by Disease Status:")
for col in ['age', 'thalach', 'oldpeak', 'trestbps', 'chol']:
    no_d = df[df['target_binary']==0][col].mean()
    yes_d = df[df['target_binary']==1][col].mean()
    print(f"  {col:10s} | No Disease: {no_d:.1f} | Disease: {yes_d:.1f}")

# ── 5. Correlation with Target ───────────────────────────────
print(f"\nCorrelation with Target Variable:")
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
corrs = df[numeric_cols + ['target_binary']].corr()['target_binary'].drop('target_binary')
for feat, corr in corrs.sort_values(key=abs, ascending=False).items():
    print(f"  {feat}: {corr:.3f}")

# ── 6. Disease Rate by Chest Pain Type ───────────────────────
print(f"\nDisease Rate by Chest Pain Type:")
cp_map = {1:'Typical Angina', 2:'Atypical Angina',
          3:'Non-anginal Pain', 4:'Asymptomatic'}
for cp, name in cp_map.items():
    rate = df[df['cp']==cp]['target_binary'].mean()*100
    n = len(df[df['cp']==cp])
    print(f"  {name} (n={n}): {rate:.1f}%")

# ── 7. Visualisations ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('CHD EDA — UCI Heart Disease Dataset (n=918)',
             fontsize=14, fontweight='bold')

# Target distribution
axes[0,0].bar(['No Disease','Disease'],
              df['target_binary'].value_counts().sort_index(),
              color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Target Distribution')

# Age by disease
df[df['target_binary']==0]['age'].hist(ax=axes[0,1], alpha=0.6,
    color='#2ecc71', label='No Disease', bins=15)
df[df['target_binary']==1]['age'].hist(ax=axes[0,1], alpha=0.6,
    color='#e74c3c', label='Disease', bins=15)
axes[0,1].set_title('Age Distribution by Disease')
axes[0,1].legend()

# Max HR vs Age scatter
for val, color, label in [(0,'#2ecc71','No Disease'),(1,'#e74c3c','Disease')]:
    subset = df[df['target_binary']==val]
    axes[0,2].scatter(subset['age'], subset['thalach'],
                      alpha=0.4, color=color, label=label, s=20)
axes[0,2].set_title('Max Heart Rate vs Age')
axes[0,2].set_xlabel('Age'); axes[0,2].set_ylabel('Max HR')
axes[0,2].legend()

# Correlation heatmap
numeric = ['age','trestbps','chol','thalach','oldpeak','target_binary']
corr = df[numeric].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1,0], linewidths=0.5, annot_kws={'size':8})
axes[1,0].set_title('Correlation Heatmap')

# Cholesterol boxplot
df.boxplot(column='chol', by='target_binary', ax=axes[1,1])
axes[1,1].set_title('Cholesterol by Disease Status')
axes[1,1].set_xlabel('0=No Disease, 1=Disease')

# ST Depression boxplot
df.boxplot(column='oldpeak', by='target_binary', ax=axes[1,2])
axes[1,2].set_title('ST Depression by Disease Status')
axes[1,2].set_xlabel('0=No Disease, 1=Disease')

plt.tight_layout()
plt.savefig('data/EDA_plots.png', dpi=150, bbox_inches='tight')
print("\nEDA plots saved to: data/EDA_plots.png")
plt.show()
