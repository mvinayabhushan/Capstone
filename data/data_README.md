# Data

## Source

UCI Machine Learning Repository — Heart Disease Dataset
- URL: https://archive.ics.uci.edu/dataset/45/heart+disease
- DOI: 10.24432/C52P4X
- Original contributors: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989)

The dataset combines clinical records from four centres: Cleveland (USA), Hungary, Switzerland, and VA Long Beach (USA).

## Licence

The dataset is distributed under the Creative Commons Attribution 4.0 International Licence (CC BY 4.0). Attribution to the original contributors and to the UCI Machine Learning Repository is required when reusing this data.

## Files in this folder

### `heart_disease_cleaned.csv`

The cleaned version of the UCI dataset used in this project. Produced by `scripts/01_data_cleaning.py` from the raw UCI files.

**Shape:** 918 rows × 12 columns (11 features + target)

**Cleaning steps applied (see synopsis Section 4.3 for full details):**

- 2 duplicate records removed.
- Invalid zero values in cholesterol (172 records) and resting BP (1 record) treated as missing and imputed with column medians.
- Remaining missing values (<10% per column for trestbps, chol, fbs, restecg, thalach, exang, oldpeak) imputed with column medians.
- Three columns with excessive missingness dropped: `slope` (33.6%), `ca` (66.4%), `thal` (52.8%).
- Target variable binarized: original severity levels 1–4 combined into a single "Disease" class (1), level 0 becomes "No Disease" (0).
- Final class balance: 55.3% disease / 44.7% no disease.

## Citation

If you use this dataset, please cite both the UCI repository and the original paper:

> Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Heart Disease Dataset*. University of California, Irvine. https://archive.ics.uci.edu/dataset/45/heart+disease

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
