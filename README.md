# Capstone
This is my AIML Capstone 
[README.md](https://github.com/user-attachments/files/26537511/README.md)
# Early Detection of Congenital Heart Disease (CHD) Using Machine Learning

**QM640: Data Analytics Capstone | Walsh College | April 2025 Term**
**Author:** Vinaya Bhushan M
**GitHub:** [mvinayabhushan/AIML](https://github.com/mvinayabhushan/AIML)

---

## Project Overview

This capstone project applies supervised machine learning techniques to structured clinical data for the early detection of Congenital Heart Disease (CHD). Using the UCI Heart Disease Dataset (918 records, 4 international cohorts), six ML classification models are developed, evaluated, and compared.

---

## Repository Structure

```
AIML/
├── data/
│   └── heart_disease_cleaned.csv       # Cleaned dataset (918 records, 11 features)
├── code/
│   ├── 01_data_cleaning.py             # Data loading, cleaning, preprocessing
│   ├── 02_eda_analysis.py              # Exploratory Data Analysis
│   └── 03_ml_models.py                 # ML model training, evaluation, comparison
└── README.md
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **DOI:** [10.24432/C52P4X](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Records:** 918 (after cleaning) | **Features:** 11 | **Target:** Binary (0=No Disease, 1=Disease)
- **Cohorts:** Cleveland, Hungarian, Switzerland, VA Long Beach

### Key Features

| Feature | Description | Type |
|---|---|---|
| age | Patient age (years) | Continuous |
| sex | Biological sex (1=Male, 0=Female) | Binary |
| cp | Chest pain type (1–4) | Categorical |
| trestbps | Resting blood pressure (mmHg) | Continuous |
| chol | Serum cholesterol (mg/dl) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise-induced angina | Binary |
| oldpeak | ST depression (exercise vs rest) | Continuous |
| target_binary | CHD diagnosis (1=Disease, 0=No Disease) | Binary |

---

## Research Questions

- **RQ1:** Which clinical features are the most statistically significant predictors of CHD?
- **RQ2:** Which ML algorithm achieves the highest classification performance?
- **RQ3:** Are there statistically significant differences in clinical indicators between CHD-positive and CHD-negative groups?
- **RQ4:** How does gender moderate the relationship between clinical features and CHD diagnosis?

---

## Models Evaluated

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Gradient Boosting | 81.0% | 0.834 | 0.893 |
| Random Forest | 78.8% | 0.813 | 0.893 |
| SVM | 79.3% | 0.819 | 0.885 |
| Logistic Regression | 78.8% | 0.812 | 0.882 |
| KNN | 79.3% | 0.819 | 0.875 |
| Decision Tree | 76.1% | 0.794 | 0.813 |

**Best Model: Gradient Boosting** — Accuracy: 81.0%, AUC-ROC: 0.893

---

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## How to Run

```bash
# Step 1: Data Cleaning
python code/01_data_cleaning.py

# Step 2: Exploratory Data Analysis
python code/02_eda_analysis.py

# Step 3: ML Model Training & Evaluation
python code/03_ml_models.py
```

---

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
