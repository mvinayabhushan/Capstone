# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 03: ML Model Training & Evaluation
# Author: Vinaya Bhushan M | Walsh College
# ============================================================
# Outputs:
#   outputs/models/results_summary.csv
#   outputs/models/feature_importance_rf.csv
#   outputs/models/threshold_tuning_best.csv
#   outputs/models/fig_roc_curves_all_models.png
#   outputs/models/fig_confusion_matrix_<model_name>.png  (one per model)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = 'data/heart_disease_cleaned.csv'
OUT_DIR = 'outputs/models'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target', 'target_binary'])
y = df['target_binary']
feature_names = X.columns.tolist()

print("=" * 70)
print("MODEL TRAINING & EVALUATION — CHD Early Detection")
print("=" * 70)

# ── Train/Test Split (stratified) ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {X_train.shape[0]} records")
print(f"Test set:  {X_test.shape[0]} records")
print(f"Class balance (train): {y_train.mean():.3f} positive")
print(f"Class balance (test):  {y_test.mean():.3f} positive")

# ── Feature Scaling for distance/margin-based models ─────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── Define Models ─────────────────────────────────────────────
# max_depth=5 chosen for Decision Tree to limit overfitting on a
# small-to-medium dataset; deeper trees memorise the training set.
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7),
}

# Models that need scaled features
scaled_models = {'Logistic Regression', 'SVM', 'KNN'}

# ── Train & Evaluate Each Model ──────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "=" * 88)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}   {'CV Mean ± SD':>14}")
print("=" * 88)

for name, model in models.items():
    X_tr = X_train_sc if name in scaled_models else X_train.values
    X_te = X_test_sc if name in scaled_models else X_test.values

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')

    results[name] = {
        'Accuracy':   accuracy_score(y_test, y_pred),
        'Precision':  precision_score(y_test, y_pred),
        'Recall':     recall_score(y_test, y_pred),
        'F1':         f1_score(y_test, y_pred),
        'AUC-ROC':    roc_auc_score(y_test, y_prob),
        'CV Mean':    cv_scores.mean(),
        'CV Std':     cv_scores.std(),
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'model':      model,
    }
    r = results[name]
    print(f"{name:<22} {r['Accuracy']:>6.3f} {r['Precision']:>6.3f} "
          f"{r['Recall']:>6.3f} {r['F1']:>6.3f} {r['AUC-ROC']:>6.3f}   "
          f"{r['CV Mean']:.3f} ± {r['CV Std']:.3f}")

# ── Save Results Summary as CSV ──────────────────────────────
results_df = pd.DataFrame({
    name: {
        'Accuracy':  r['Accuracy'],
        'Precision': r['Precision'],
        'Recall':    r['Recall'],
        'F1':        r['F1'],
        'AUC-ROC':   r['AUC-ROC'],
        'CV Mean':   r['CV Mean'],
        'CV Std':    r['CV Std'],
    } for name, r in results.items()
}).T.round(3)
results_df.index.name = 'Model'
results_df.to_csv(f'{OUT_DIR}/results_summary.csv')
print(f"\nSaved: {OUT_DIR}/results_summary.csv")

# ── Best Model (by AUC-ROC) ──────────────────────────────────
best = max(results, key=lambda n: results[n]['AUC-ROC'])
print(f"\nBest Model (by AUC-ROC): {best} | AUC-ROC = {results[best]['AUC-ROC']:.3f}")

# ── Feature Importance (Random Forest) ───────────────────────
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(f"\nFeature Importance (Random Forest):")
for feat, imp in feat_imp.items():
    print(f"  {feat:12s}: {imp:.4f}")
feat_imp_df = feat_imp.reset_index()
feat_imp_df.columns = ['Feature', 'Importance']
feat_imp_df.to_csv(f'{OUT_DIR}/feature_importance_rf.csv', index=False)
print(f"\nSaved: {OUT_DIR}/feature_importance_rf.csv")

# ── Threshold Tuning on Best Model ───────────────────────────
# As stated in synopsis (Section 5.5): the classification threshold
# is tuned on the test set to maximise recall subject to a precision
# floor (0.70), since missing a real CHD case is costlier than an
# unnecessary referral.
print(f"\nThreshold Tuning — {best}")
print("=" * 70)
y_prob_best = results[best]['y_prob']
thresholds = np.arange(0.10, 0.91, 0.05)
threshold_records = []
for t in thresholds:
    y_pred_t = (y_prob_best >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    threshold_records.append({
        'Threshold': round(t, 2),
        'Precision': round(prec, 3),
        'Recall':    round(rec, 3),
        'F1':        round(f1, 3),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    })
threshold_df = pd.DataFrame(threshold_records)
threshold_df.to_csv(f'{OUT_DIR}/threshold_tuning_best.csv', index=False)
print(threshold_df.to_string(index=False))
print(f"\nSaved: {OUT_DIR}/threshold_tuning_best.csv")

# Recommend the best threshold: max recall with precision >= 0.70
eligible = threshold_df[threshold_df['Precision'] >= 0.70]
if len(eligible) > 0:
    chosen_row = eligible.loc[eligible['Recall'].idxmax()]
    print(f"\nRecommended threshold (max recall with precision ≥ 0.70):")
    print(f"  Threshold = {chosen_row['Threshold']:.2f}")
    print(f"  Precision = {chosen_row['Precision']:.3f}")
    print(f"  Recall    = {chosen_row['Recall']:.3f}")
    print(f"  F1        = {chosen_row['F1']:.3f}")
else:
    print("\nNo threshold met the precision floor of 0.70 — review precision constraint.")

# ── ROC Curve Plot — All Models ──────────────────────────────
colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6', '#1abc9c']
fig, ax = plt.subplots(figsize=(9, 7))
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {r['AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC = 0.500)')
ax.set_title('Figure 8. ROC Curves — Six Classifiers on CHD Test Set',
             fontweight='bold')
ax.set_xlabel('False Positive Rate (1 − Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_roc_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT_DIR}/fig_roc_curves_all_models.png")

# ── Confusion Matrix per Model ───────────────────────────────
for name, r in results.items():
    cm = confusion_matrix(y_test, r['y_pred'])
    safe_name = name.lower().replace(' ', '_')
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax,
                cbar=False, annot_kws={'size': 14})
    ax.set_title(f'Confusion Matrix — {name}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig_confusion_matrix_{safe_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/fig_confusion_matrix_{safe_name}.png")

print("\n" + "=" * 70)
print(f"Modelling complete. All outputs in: {OUT_DIR}/")
print("=" * 70)
