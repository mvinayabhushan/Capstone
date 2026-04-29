# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 04: Statistical Tests + Gender-Stratified Evaluation
# Author: Vinaya Bhushan M | Walsh College
# ============================================================
# Fulfils synopsis Sections 5.1, 5.3, 5.4:
#   RQ1 — Feature significance (t-tests, chi-square, effect sizes)
#   RQ3 — Group differences (Levene + t-test or Welch + Cohen's d)
#   RQ4 — Gender moderation (chi-square + gender-stratified evaluation)
#
# Outputs:
#   outputs/stats/rq1_continuous_tests.csv
#   outputs/stats/rq1_categorical_tests.csv
#   outputs/stats/rq3_group_differences.csv
#   outputs/stats/rq4_gender_prevalence.csv
#   outputs/stats/rq4_gender_stratified_performance.csv
#   outputs/stats/fig_rq4_roc_by_gender.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                              precision_score, recall_score, f1_score)

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = 'data/heart_disease_cleaned.csv'
OUT_DIR = 'outputs/stats'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper functions ─────────────────────────────────────────
def cohens_d(x, y):
    """Cohen's d for independent samples (pooled SD)."""
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd

def cramers_v(contingency_table):
    """Cramer's V for chi-square contingency tables."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))

def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return 'negligible'
    if ad < 0.5: return 'small'
    if ad < 0.8: return 'medium'
    return 'large'

def interpret_v(v):
    if v < 0.1: return 'negligible'
    if v < 0.3: return 'small'
    if v < 0.5: return 'medium'
    return 'large'

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 70)
print("STATISTICAL TESTS — CHD Early Detection")
print("=" * 70)
print(f"\nDataset: {df.shape[0]} records, {df.shape[1]} columns")

# ============================================================
# RQ1 — FEATURE SIGNIFICANCE (Synopsis Section 5.1)
# ============================================================
print("\n" + "=" * 70)
print("RQ1 — Feature Significance")
print("=" * 70)

# RQ1.a — Continuous features: t-test + Cohen's d
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
group0 = df[df['target_binary'] == 0]
group1 = df[df['target_binary'] == 1]

cont_records = []
print("\nContinuous features — Independent samples t-test:")
print(f"{'Feature':<12} {'Mean (No)':>12} {'Mean (Yes)':>12} {'t':>8} {'p-value':>10} {'Cohens d':>12} {'Effect':>12}")
print("-" * 80)
for col in continuous_features:
    x = group0[col].dropna()
    y = group1[col].dropna()
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)  # Welch by default; Levene-driven choice handled in RQ3
    d = cohens_d(x, y)
    effect = interpret_d(d)
    cont_records.append({
        'Feature': col,
        'Mean_NoDisease': round(x.mean(), 3),
        'Mean_Disease':   round(y.mean(), 3),
        't_statistic':    round(t_stat, 3),
        'p_value':        round(p_val, 6),
        'Cohens_d':       round(d, 3),
        'Effect_Size':    effect,
        'Significant_at_0.05': p_val < 0.05,
    })
    print(f"{col:<12} {x.mean():>12.2f} {y.mean():>12.2f} {t_stat:>8.3f} {p_val:>10.6f} {d:>12.3f} {effect:>12}")

pd.DataFrame(cont_records).to_csv(f'{OUT_DIR}/rq1_continuous_tests.csv', index=False)
print(f"\nSaved: {OUT_DIR}/rq1_continuous_tests.csv")

# RQ1.b — Categorical features: chi-square + Cramer's V
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang']
cat_records = []
print("\nCategorical features — Chi-square test of independence:")
print(f"{'Feature':<12} {'Chi2':>10} {'df':>4} {'p-value':>10} {'Cramers V':>12} {'Effect':>12}")
print("-" * 70)
for col in categorical_features:
    contingency = pd.crosstab(df[col], df['target_binary'])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    v = cramers_v(contingency)
    effect = interpret_v(v)
    cat_records.append({
        'Feature': col,
        'Chi2_statistic':    round(chi2, 3),
        'Degrees_of_Freedom': dof,
        'p_value':           round(p_val, 6),
        'Cramers_V':         round(v, 3),
        'Effect_Size':       effect,
        'Significant_at_0.05': p_val < 0.05,
    })
    print(f"{col:<12} {chi2:>10.3f} {dof:>4} {p_val:>10.6f} {v:>12.3f} {effect:>12}")

pd.DataFrame(cat_records).to_csv(f'{OUT_DIR}/rq1_categorical_tests.csv', index=False)
print(f"\nSaved: {OUT_DIR}/rq1_categorical_tests.csv")

# ============================================================
# RQ3 — GROUP DIFFERENCES (Synopsis Section 5.3)
# Levene first; then t-test or Welch's t-test based on result.
# Variables: trestbps, chol, thalach
# ============================================================
print("\n" + "=" * 70)
print("RQ3 — Group Differences (Resting BP, Cholesterol, Max HR)")
print("=" * 70)
rq3_features = ['trestbps', 'chol', 'thalach']
rq3_records = []
print(f"\n{'Feature':<12} {'Levene p':>10} {'Equal Var':>11} {'Test':>10} {'t':>8} {'p-value':>10} {'Cohens d':>12}")
print("-" * 85)
for col in rq3_features:
    x = group0[col].dropna()
    y = group1[col].dropna()

    # Step 1: Levene's test for equal variances
    levene_stat, levene_p = stats.levene(x, y)
    equal_var = levene_p >= 0.05  # If p >= 0.05, variances assumed equal

    # Step 2: Choose test based on Levene's result
    test_used = "Student's t" if equal_var else "Welch's t"
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=equal_var)

    # Step 3: Cohen's d
    d = cohens_d(x, y)

    rq3_records.append({
        'Feature': col,
        'Levene_p':       round(levene_p, 6),
        'Equal_Variance': equal_var,
        'Test_Used':      test_used,
        't_statistic':    round(t_stat, 3),
        'p_value':        round(p_val, 6),
        'Cohens_d':       round(d, 3),
        'Effect_Size':    interpret_d(d),
        'Significant_at_0.05': p_val < 0.05,
    })
    print(f"{col:<12} {levene_p:>10.4f} {str(equal_var):>11} {test_used:>10} {t_stat:>8.3f} {p_val:>10.6f} {d:>12.3f}")

pd.DataFrame(rq3_records).to_csv(f'{OUT_DIR}/rq3_group_differences.csv', index=False)
print(f"\nSaved: {OUT_DIR}/rq3_group_differences.csv")

# ============================================================
# RQ4 — GENDER MODERATION (Synopsis Section 5.4)
# Part A: Chi-square — does CHD prevalence differ by gender?
# Part B: Train Gradient Boosting; evaluate separately for males and females.
# ============================================================
print("\n" + "=" * 70)
print("RQ4 — Gender Moderation")
print("=" * 70)

# Part A: Chi-square for gender x CHD
gender_table = pd.crosstab(df['sex'], df['target_binary'])
gender_table.index = ['Female', 'Male']
gender_table.columns = ['No Disease', 'Disease']
print(f"\nGender × CHD contingency table:")
print(gender_table)

chi2, p_val, dof, _ = stats.chi2_contingency(gender_table)
v = cramers_v(gender_table)
prev_female = gender_table.loc['Female', 'Disease'] / gender_table.loc['Female'].sum()
prev_male = gender_table.loc['Male', 'Disease'] / gender_table.loc['Male'].sum()

print(f"\nCHD prevalence — Female: {prev_female*100:.1f}%")
print(f"CHD prevalence — Male:   {prev_male*100:.1f}%")
print(f"Chi-square = {chi2:.3f}, df = {dof}, p = {p_val:.6f}")
print(f"Cramer's V = {v:.3f} ({interpret_v(v)})")

pd.DataFrame([{
    'Female_CHD_Prevalence': round(prev_female, 3),
    'Male_CHD_Prevalence':   round(prev_male, 3),
    'Chi2':                  round(chi2, 3),
    'Degrees_of_Freedom':    dof,
    'p_value':               round(p_val, 6),
    'Cramers_V':             round(v, 3),
    'Effect_Size':           interpret_v(v),
    'Significant_at_0.05':   p_val < 0.05,
}]).to_csv(f'{OUT_DIR}/rq4_gender_prevalence.csv', index=False)
print(f"\nSaved: {OUT_DIR}/rq4_gender_prevalence.csv")

# Part B: Gender-stratified evaluation using Gradient Boosting
print("\nGender-stratified evaluation — Gradient Boosting:")
X = df.drop(columns=['target', 'target_binary'])
y = df['target_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Identify male/female test indices
test_df = X_test.copy()
test_df['y_true'] = y_test.values
test_df['y_pred'] = y_pred
test_df['y_prob'] = y_prob

gender_perf = []
roc_data = {}
for sex_code, label in [(0, 'Female'), (1, 'Male')]:
    sub = test_df[test_df['sex'] == sex_code]
    if len(sub) == 0 or sub['y_true'].nunique() < 2:
        print(f"  {label}: insufficient test samples or single class — skipping")
        continue
    cm = confusion_matrix(sub['y_true'], sub['y_pred'])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    auc = roc_auc_score(sub['y_true'], sub['y_prob'])
    prec = precision_score(sub['y_true'], sub['y_pred'], zero_division=0)
    rec = recall_score(sub['y_true'], sub['y_pred'], zero_division=0)
    f1 = f1_score(sub['y_true'], sub['y_pred'], zero_division=0)

    fpr, tpr, _ = roc_curve(sub['y_true'], sub['y_prob'])
    roc_data[label] = (fpr, tpr, auc)

    gender_perf.append({
        'Group':     label,
        'N_Test':    len(sub),
        'Precision': round(prec, 3),
        'Recall':    round(rec, 3),
        'F1':        round(f1, 3),
        'AUC_ROC':   round(auc, 3),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    })
    print(f"  {label}: n={len(sub)}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

pd.DataFrame(gender_perf).to_csv(f'{OUT_DIR}/rq4_gender_stratified_performance.csv', index=False)
print(f"\nSaved: {OUT_DIR}/rq4_gender_stratified_performance.csv")

# Gender-stratified ROC plot
if len(roc_data) == 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Female': '#e91e63', 'Male': '#2196f3'}
    for label, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, color=colors[label], lw=2,
                label=f'{label} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_title('Figure 8. ROC Curves by Gender — Gradient Boosting',
                 fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_rq4_roc_by_gender.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/fig_rq4_roc_by_gender.png")

print("\n" + "=" * 70)
print(f"Statistical tests complete. All outputs in: {OUT_DIR}/")
print("=" * 70)
