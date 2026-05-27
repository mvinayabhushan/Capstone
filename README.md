# Early Detection of Coronary Heart Disease Using Machine Learning

**QM 640: Data Analytics Capstone — Walsh College**  
**Author:** Vinaya Bhushan M  
**Term:** Fall 2025, Term 3  
**Mentor:** Dr. Vikas S

---

## Project Overview

This project develops and evaluates a non-invasive machine learning screening tool for early detection of Coronary Heart Disease (CHD) using 11 routine clinical features collected during outpatient visits. Six supervised classifiers are compared using stratified cross-validation, hypothesis testing validates feature-level group differences, and a gender-stratified fairness audit with threshold tuning addresses the male-skewed dataset composition.

**Best model:** Support Vector Machine (RBF kernel, C = 0.5)  
- Cross-validation recall: **0.8646**  
- Test-set recall: **0.8627** | AUC: **0.8846**  
- Female recall at tuned threshold (0.35): **0.875** (up from 0.500 at default 0.50)

---

## Repository Structure

```
Capstone/
├── data/
│   ├── raw/                        # Original UCI source files (4 institutions)
│   │   ├── processed.cleveland.data
│   │   ├── processed.hungarian.data
│   │   ├── processed.switzerland.data
│   │   └── processed.va.data
│   └── heart_disease_cleaned.csv   # Cleaned dataset (918 records, 12 columns)
│
├── scripts/
│   ├── 01_data_cleaning.py         # Data loading, cleaning, reproducibility assertions
│   ├── 02_eda_analysis.py          # Exploratory data analysis & visualisations
│   ├── 03_ml_models.py             # Model training, CV evaluation, threshold tuning
│   └── 04_statistical_tests.py    # Hypothesis testing (RQ1, RQ3, RQ4)
│
├── outputs/
│   ├── eda/                        # EDA figures and descriptive stats CSVs
│   ├── models/                     # Model comparison figures, confusion matrices, CSVs
│   └── stats/                      # Statistical test results (RQ1–RQ4 CSVs + figures)
│
├── docs/
│   ├── QM640_Final_Report_Vinaya_Bhushan_M.pdf   # Final report
│   ├── QM640_Interim_Report_Vinaya_Bhushan_M.pdf # Interim report
│   └── CHD_Synopsis.pdf                          # Project synopsis
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Data Source

The original raw data is the **UCI Heart Disease Dataset** (Janosi et al., 1989; Dua & Graff, 2019):  
https://archive.ics.uci.edu/ml/datasets/heart+disease

- Institutions: Cleveland, Hungarian, Zurich, Basel, VA Long Beach (1981–1987)
- N = 918 unique records after deduplication
- 11 clinical features, binary CHD outcome (55.3% positive prevalence)
- No personally identifiable information; IRB exemption applies

---

## How to Run

### 1. Install dependencies (one-time)
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline stages in order
```bash
python scripts/01_data_cleaning.py
python scripts/02_eda_analysis.py
python scripts/03_ml_models.py
python scripts/04_statistical_tests.py
```

> **Note:** `data/heart_disease_cleaned.csv` is included. Run `01_data_cleaning.py` only if you want to regenerate it from the raw source files in `data/raw/`.

### Requirements
- Python 3.11+
- CPU only — no GPU required

---

## Research Questions

| RQ | Question | Method | Script |
|----|----------|--------|--------|
| RQ1 | Which clinical features are most associated with CHD and what are their effect sizes? | Random Forest feature importance + Welch's t-test with Cohen's d | `03_ml_models.py`, `04_statistical_tests.py` |
| RQ2 | Which of six classifiers performs best for non-invasive CHD screening? | Stratified 5-fold CV (recall primary, AUC secondary) | `03_ml_models.py` |
| RQ3 | Are there statistically significant differences in continuous clinical indicators between CHD-positive and CHD-negative groups? | Welch's two-sample t-test | `04_statistical_tests.py` |
| RQ4 | Does the best model perform equitably across gender subgroups? | Subgroup recall comparison + decision threshold tuning | `04_statistical_tests.py` |

---

## Key Results

| Model | CV Recall | CV AUC | Test Recall | Test AUC |
|-------|-----------|--------|-------------|----------|
| **SVM (RBF, C=0.5)** | **0.8646** | 0.8893 | **0.8627** | **0.8846** |
| Random Forest | 0.8547 | **0.8977** | 0.8431 | 0.8699 |
| Gradient Boosting | 0.8350 | 0.8901 | 0.8235 | 0.8712 |
| Logistic Regression | 0.8350 | 0.8823 | 0.8039 | 0.8754 |
| Decision Tree | 0.7972 | 0.7897 | 0.7843 | 0.7820 |
| KNN | 0.7913 | 0.8396 | 0.7647 | 0.8284 |

**Gender Fairness (SVM):**
- Default threshold (0.50): Male recall = 0.8936, Female recall = 0.500 → 39 pp gap
- Tuned threshold (0.35): Male recall = 0.9362, Female recall = 0.875 → 6 pp gap ✅

---

## License

Released under the **MIT License**. The UCI Heart Disease Dataset is freely redistributable for educational and research purposes.

---

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Dua, D., & Graff, C. (2019). UCI machine learning repository. University of California, Irvine. https://archive.ics.uci.edu/ml/
