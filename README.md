# Early Detection of Coronary Heart Disease Using Machine Learning

**QM 640: Data Analytics Capstone — Walsh College**  
**Author:** Vinaya Bhushan M  
**Term:** Fall 2025, Term 3  
**Mentor:** Dr. Vikas S

---

## Project Overview

This project develops and evaluates a non-invasive machine learning screening tool for early detection of Coronary Heart Disease (CHD) using 11 routine clinical features collected during outpatient visits. Six supervised classifiers are compared using stratified cross-validation, hypothesis testing validates feature-level group differences, and a gender-stratified fairness audit with threshold tuning addresses the male-skewed dataset.

**Best model:** Support Vector Machine (RBF kernel, C = 0.5)  
- Cross-validation recall: **0.8646**  
- Test-set recall: **0.8627** | AUC: **0.8846**  
- Female recall at tuned threshold (0.35): **0.875** (up from 0.500 at default 0.50)

---

## Repository Contents

| File | Description |
|------|-------------|
| `heart_disease_cleaned.csv` | Cleaned dataset — 918 deduplicated patient records derived from the UCI Heart Disease Dataset |
| `01_data_cleaning.py` | Stage 1 — raw data ingestion, duplicate removal, missing value handling |
| `02_eda_analysis.py` | Stage 2 — exploratory data analysis, distribution plots, correlation analysis |
| `03_ml_models.py` | Stage 3 — model training, cross-validation, evaluation, and fairness audit |

---

## Data Source

The original raw data is the **UCI Heart Disease Dataset** (Janosi et al., 1989; Dua & Graff, 2019), freely available at:  
https://archive.ics.uci.edu/ml/datasets/heart+disease

- Institutions: Cleveland, Hungarian, Zurich, Basel, VA Long Beach (1981–1987)  
- N = 918 unique records after deduplication  
- 11 clinical features, binary CHD outcome (55.3% positive prevalence)  
- No personally identifiable information; IRB exemption applies

---

## How to Run

### 1. Install dependencies (one-time)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ucimlrepo
```

### 2. Run the pipeline stages in order
```bash
python 01_data_cleaning.py
python 02_eda_analysis.py
python 03_ml_models.py
```

> **Note:** `heart_disease_cleaned.csv` is included in the repo. If you want to regenerate it from scratch, `01_data_cleaning.py` downloads the raw data from the UCI repository automatically.

### Requirements
- Python 3.11+
- CPU only — no GPU required

---

## Research Questions

| RQ | Question | Method |
|----|----------|--------|
| RQ1 | Which clinical features are most associated with CHD and what are their effect sizes? | Random Forest feature importance + Welch's t-test with Cohen's d |
| RQ2 | Which of six classifiers performs best for non-invasive CHD screening? | Stratified 5-fold cross-validation (recall primary, AUC secondary) |
| RQ3 | Are there statistically significant differences in continuous clinical indicators between CHD-positive and CHD-negative groups? | Welch's two-sample t-test |
| RQ4 | Does the best model perform equitably across gender subgroups? | Subgroup recall comparison + decision threshold tuning |

---

## License

This project is released under the **MIT License** and is governed by Walsh College academic integrity policies. The UCI Heart Disease Dataset is freely redistributable for educational and research purposes.

---

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Dua, D., & Graff, C. (2019). UCI machine learning repository. University of California, Irvine. https://archive.ics.uci.edu/ml/
