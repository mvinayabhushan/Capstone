# Early Detection of Coronary Heart Disease Using Machine Learning

**QM640 Data Analytics Capstone — Walsh College, Fall 2025 (Term 3)**
Author: Vinaya Bhushan M
Mentor: Vikas S

## About the Project

This capstone looks at whether a supervised machine learning model trained on routine clinical data can flag patients at high risk of Coronary Heart Disease (CHD) early, so that primary care physicians and cardiologists can prioritize them for further testing without having to wait for invasive diagnostics.

The idea is simple. Most of the data that would help catch CHD early (blood pressure, cholesterol, max heart rate, resting ECG) already sits in routine patient records. This project explores whether that data is enough, on its own, to build a screening tool that is reliable enough to support clinical decisions.

## Dataset

The project uses the **UCI Machine Learning Repository Heart Disease Dataset** (Janosi et al., 1989; Dua & Graff, 2019), combining clinical records from four centres: Cleveland, Hungary, Switzerland, and VA Long Beach. After cleaning, 918 records and 11 features are retained, with a binary target (disease / no disease).

- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **DOI:** 10.24432/C52P4X
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

No Kaggle data is used, in line with course policy.

## Research Questions

1. **RQ1 — Feature significance.** Which clinical features best predict CHD, and how much do they contribute to model performance?
2. **RQ2 — Model comparison.** Which algorithm among Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, and KNN performs best for CHD classification?
3. **RQ3 — Group differences.** Do CHD-positive and CHD-negative patients differ significantly on resting blood pressure, cholesterol, and maximum heart rate?
4. **RQ4 — Gender moderation.** Does gender affect the relationship between clinical features and CHD, and does the dataset's 79% male skew hurt the model's fairness?

## Repository Structure

```
Capstone/
├── README.md                      This file
├── LICENSE                        MIT licence for the code
├── requirements.txt               Python package versions used
├── .gitignore                     Files Git should ignore
├── data/
│   ├── README.md                  Notes on the data source and licence
│   └── heart_disease_cleaned.csv  918 cleaned records, 11 features
├── scripts/
│   ├── 01_data_cleaning.py        Raw UCI to cleaned CSV
│   ├── 02_eda_analysis.py         Exploratory data analysis
│   └── 03_ml_models.py            Model training and evaluation
└── docs/
    └── CHD_Synopsis.pdf           Final synopsis submitted for QM640
```

## How to Run

Python 3.10 or later is recommended.

```bash
# 1. Clone the repo
git clone https://github.com/mvinayabhushan/Capstone.git
cd Capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run scripts in order
python scripts/01_data_cleaning.py
python scripts/02_eda_analysis.py
python scripts/03_ml_models.py
```

## Documents

The full capstone synopsis, including methodology, sample size calculations, analytic approach, and recommendations, is in `docs/CHD_Synopsis.pdf`.

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`). The underlying UCI dataset is distributed under CC BY 4.0 and its own terms apply to the data itself.
