# ============================================================
# QM640 Capstone: CHD Early Detection
# Script 03: ML Model Training & Evaluation
# Author: Vinaya Bhushan M | Walsh College
# ============================================================

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

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv('data/heart_disease_cleaned.csv')
X = df.drop(columns=['target', 'target_binary'])
y = df['target_binary']

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Feature Scaling ───────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define Models ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7),
}

# Scaled models (distance/gradient-based)
scaled_models = ['Logistic Regression', 'SVM', 'KNN']

# ── Train & Evaluate ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "="*70)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV':>12}")
print("="*70)

for name, model in models.items():
    X_tr = X_train_sc if name in scaled_models else X_train.values
    X_te = X_test_sc  if name in scaled_models else X_test.values

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')

    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_prob),
        'CV Mean':   cv_scores.mean(),
        'CV Std':    cv_scores.std(),
        'y_pred': y_pred, 'y_prob': y_prob, 'model': model,
    }
    r = results[name]
    print(f"{name:<22} {r['Accuracy']:>6.3f} {r['Precision']:>6.3f} "
          f"{r['Recall']:>6.3f} {r['F1']:>6.3f} {r['AUC-ROC']:>6.3f} "
          f"{r['CV Mean']:.3f}±{r['CV Std']:.3f}")

# ── Best Model ────────────────────────────────────────────────
best = max(results, key=lambda n: results[n]['AUC-ROC'])
print(f"\nBest Model: {best} | AUC-ROC: {results[best]['AUC-ROC']:.3f}")

# ── Feature Importance (Random Forest) ───────────────────────
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nTop Features (Random Forest):")
for feat, imp in feat_imp.items():
    print(f"  {feat:12s}: {imp:.4f}")

# ── ROC Curve Plot ────────────────────────────────────────────
colors = ['#3498db','#e67e22','#2ecc71','#e74c3c','#9b59b6','#1abc9c']
plt.figure(figsize=(9, 6))
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC={r['AUC-ROC']:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.title('ROC Curves — CHD Early Detection Models', fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/ROC_curves.png', dpi=150, bbox_inches='tight')
print("\nROC curve saved to: data/ROC_curves.png")

# ── Confusion Matrix — Best Model ─────────────────────────────
cm = confusion_matrix(y_test, results[best]['y_pred'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])
plt.title(f'Confusion Matrix — {best}', fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('data/confusion_matrix_best.png', dpi=150, bbox_inches='tight')
print(f"Confusion matrix saved to: data/confusion_matrix_best.png")
plt.show()
