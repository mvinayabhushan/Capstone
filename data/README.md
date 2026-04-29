# Data Folder

This folder holds the raw UCI files (in `raw/`) and the cleaned dataset produced by `scripts/01_data_cleaning.py`.

## Source

UCI Machine Learning Repository — Heart Disease Dataset
- Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
- File directory: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
- DOI: 10.24432/C52P4X
- Original contributors: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989)

The dataset combines clinical records from four centres: Cleveland (USA), Hungary, Switzerland, and VA Long Beach (USA).

## Licence

Distributed under the Creative Commons Attribution 4.0 International Licence (CC BY 4.0). Attribution to the original contributors and to the UCI Machine Learning Repository is required when reusing this data.

## Files in this Folder

| File | Description |
|------|-------------|
| `raw/processed.cleveland.data` | 303 records — Cleveland Clinic Foundation, USA |
| `raw/processed.hungarian.data` | 294 records — Hungarian Institute of Cardiology, Budapest |
| `raw/processed.switzerland.data` | 123 records — University Hospital, Zurich |
| `raw/processed.va.data` | 200 records — VA Medical Center, Long Beach, USA |
| `heart_disease_cleaned.csv` | 918 cleaned records, 11 features + binary target |

### `heart_disease_cleaned.csv`

The cleaned version of the UCI dataset used in this project. Produced by `scripts/01_data_cleaning.py` from the four raw UCI files.

**Shape:** 918 rows × 12 columns (11 features + target + binarised target)

**Cleaning steps applied** (see synopsis Section 4.3 for full details):

- 2 duplicate records removed.
- Invalid zero values in cholesterol and resting BP treated as missing and imputed with column medians.
- Remaining missing values (<10% per column for `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`) imputed with column medians.
- Three columns with excessive missingness dropped: `slope` (33.6%), `ca` (66.4%), `thal` (52.8%).
- Target variable binarised: original severity levels 1–4 combined into a single "Disease" class (1), level 0 becomes "No Disease" (0).
- Final class balance: 55.3% disease / 44.7% no disease.

## Reproducing the Raw Download

```bash
cd data/raw
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data
```

All four raw files share a 14-column structure with `?` denoting missing values.

## Citation

If you use this dataset, please cite both the UCI repository and the original paper:

> Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Heart Disease Dataset*. University of California, Irvine. https://archive.ics.uci.edu/dataset/45/heart+disease

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
